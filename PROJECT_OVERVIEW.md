# Robust Portfolio Optimization – Aperçu Complet

Ce document résume les fonctionnalités du projet et explique l’écart de performances observé entre les stratégies « box » et « budgeted ».

## 1. Objectif général

L’application construit et évalue des portefeuilles robustes en utilisant un backtest roulant sur des rendements historiques :
- chargement de prix (Yahoo Finance ou CSV locaux), transformation en rendements logarithmiques ;
- estimation récursive des moyennes, covariances et incertitudes ;
- allocation via différentes approches (Markowitz, robustes boîte/ellipsoïde/budget-budget) ;
- évaluation hors-échantillon, mesures de performance et visualisations ;
- simulation de scénarios de krach personnalisés.

`demo.py` sert d’entrée principale pour un backtest standard, tandis que `crash_demo.py` injecte des chocs sur mesure et analyse les portefeuilles autour des dates critiques.

## 2. Flux de données

| Étape | Module | Description |
|-------|--------|-------------|
| Import des prix | `src/data_loader.py` | Télécharge ou lit les cours ajustés, nettoie et aligne les séries. |
| Rendements | `to_log_returns` | Convertit les prix en rendements logarithmiques et supprime la 1re ligne `NaN`. |
| Fenêtrage | `rolling_backtest` (`src/backtest.py`) | Découpe l’historique en fenêtres d’entraînement `T_train` et d’évaluation `T_test`. |

Les constantes par défaut (tickers, dates, tailles de fenêtres, paramètres de robustesse) sont dans `src/config.py`.

## 3. Estimation des paramètres

`src/estimators.py` fournit :
- `estimate_mean_cov` : moyenne vectorielle et covariance Ledoit–Wolf (ou échantillonnale) rendues définies positives via `ensure_pd`.
- `garch_vol_forecast` : prévisions individuelles de volatilité (arch/GARCH en option).
- `covariance_of_mean` : covariance de la moyenne via bootstrap par blocs, clé pour quantifier l’incertitude.

Ces estimations alimentent les solveurs robustes à chaque rebalancement.

## 4. Optimisation des portefeuilles

`src/optimizers.py` définit quatre stratégies long-only :
- `markowitz` : utilité quadratique classique (rendement attendu – λ × variance).
- `robust_box` : incertitude bornée composante par composante, pénalise la somme `|w_i|·δ_i`.
- `robust_ellipsoid` : incertitude ellipsoïdale `ρ‖Mᵀw‖₂` avec `M` issu de la covariance de la moyenne.
- `robust_budgeted` : modèle Bertsimas–Sim avec budget `Γ`, variables auxiliaires (`t`, `s_i`) pour activer jusqu’à `Γ` pires chocs.

Toutes les optimisations passent par `_solve_with_fallback` qui enchaîne ECOS, OSQP puis SCS et récupère la meilleure solution disponible.

## 5. Backtest roulant

`rolling_backtest` :
1. Sélectionne une fenêtre d’entraînement et d’évaluation.
2. Calcule `mu`, `Σ`, la covariance de la moyenne `Cov_mu`.
3. Évalue l’incertitude `delta = sqrt(diag(Cov_mu))`, éventuellement modulée par la volatilité GARCH relative et un facteur `delta_mult`.
4. Résout chaque stratégie, normalise les poids (`normalize_weights`), applique les rendements sur la période test.
5. Stocke retours et poids journaliers pour analyse ultérieure.

Le backtest peut activer/désactiver shrinkage, GARCH, et sélectionner un sous-ensemble de stratégies.

## 6. Évaluation et visualisation

- `src/metrics.py` : calcul de CAGR, volatilité annualisée, Sharpe, Sortino, drawdown, CVaR et concentration (HHI), avec tests t appariés via `compile_performance_table`.
- `src/visualization.py` : graphiques NAV cumulée (lin/log), Sharpe roulant, boxplots, nuages rendement-volatilité, heatmaps de poids, ainsi que des visualisations dédiées aux scénarios de krach.

`demo.py` orchestre ces sorties (console + figures Matplotlib).

## 7. Analyse de scénarios de krach

`crash_demo.py` et `src/scenario_analysis.py` permettent :
- de définir des chocs déterministes (`CrashScenario`) : date, sévérité (rendement simple), durée ;
- d’injecter un choc log-return uniforme (`apply_crash_shock`) ;
- d’extraire les poids au moment du choc, reconstituer la NAV et le front efficient, et de générer les graphiques correspondants.

Deux scénarios par défaut sont fournis (`Flash2015` et `Covid2020`), mais l’utilisateur peut en créer d’autres via l’argument CLI `--crash`.

## 8. Différence entre les stratégies « box » et « budgeted »

Historiquement, ces stratégies affichaient des rendements nettement inférieurs pour deux raisons principales :

1. **Échelle de l’incertitude (`delta`)**  
   - Initialement, `delta` était basé sur la volatilité quotidienne (`garch_vol_forecast`).  
   - Cette volatilité (~1–2 %) est largement supérieure à l’incertitude sur la moyenne (≈0,05 % par jour).  
   - Dans `robust_box`, la pénalité `Σ |w_i|·δ_i` dépassait donc très vite le terme de rendement `w·μ`.  
   - `robust_budgeted` amplifiait encore cet effet via `Γ·t + Σ s_i`, ce qui clampait les poids vers zéro.

2. **Devise des modèles robustes**  
   - Ces formulations supposent que `δ_i` décrit l’erreur sur **la moyenne** (erreur standard), pas la volatilité brute.  
   - Utiliser la mauvaise grandeur revient à surestimer la menace, exactement ce pour quoi le box et le budgeted sont sensibles.

### Correctif appliqué

Le backtest s’appuie désormais sur `sqrt(diag(Cov_mu))` (incertitude de la moyenne) et, si GARCH est activé, la volatilité ne sert qu’à moduler `δ` relativement (ratio par la moyenne des sigmas). Ainsi :
- les stratégies robustes restent conservatrices pour les actifs volatils ;
- mais elles récupèrent un rendement comparable au Markowitz, car la pénalité est calibrée à la bonne échelle ;
- `delta_mult` et `gamma` gardent leur rôle de réglage fin.

En résumé, l’écart de performance provenait d’un sur-calibrage de l’incertitude. En ramenant `δ` au niveau pertinent (erreur sur la moyenne), les modèles « box » et « budgeted » jouent leur rôle d’atténuation du risque sans annihiler la performance espérée.

## 9. Utilisation rapide

```bash
# Backtest standard
python demo.py --tickers AAPL MSFT NVDA --start 2015-01-01 --no_garch

# Simulation de crash custom
python crash_demo.py --crash MyCrash:2022-06-13:-0.18:2 --no_shrinkage
```

Adaptez les paramètres (`--train`, `--test`, `--delta_mult`, `--gamma`, etc.) pour examiner différentes hypothèses de marché.

---

Pour approfondir, consultez les modules correspondants dans `src/`. Le projet reste modulaire : importer les fonctions de `estimators`, `optimizers` ou `scenario_analysis` suffit pour les utiliser dans d’autres notebooks ou pipelines.

