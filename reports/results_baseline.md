# Step 3 - baseline results

Baselines only: no fitted team strengths. `world_cup_2018` and `world_cup_2022` are the tournament windows themselves; `validation` and `test` are the full calendar blocks they sit inside.

| model | split | n_matches | mae | rmse | poisson_deviance | mean_predicted_goals | accuracy | log_loss | rps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_league_average | validation | 717 | 1.1546 | 1.4223 | 1.3563 | 1.6964 | 0.4951 | 1.0560 | 0.2404 |
| baseline_home_away | validation | 717 | 1.1280 | 1.4043 | 1.3266 | 1.6984 | 0.4770 | 1.0323 | 0.2313 |
| baseline_form | validation | 717 | 1.0864 | 1.3636 | 1.2657 | 1.6849 | 0.4881 | 1.0297 | 0.2299 |
| baseline_league_average | test | 167 | 1.0928 | 1.3967 | 1.2847 | 1.6964 | 0.4611 | 1.0366 | 0.2445 |
| baseline_home_away | test | 167 | 1.0675 | 1.3712 | 1.2440 | 1.6362 | 0.5269 | 1.0145 | 0.2352 |
| baseline_form | test | 167 | 1.0301 | 1.3306 | 1.1931 | 1.6945 | 0.5749 | 0.9529 | 0.2141 |
| baseline_league_average | world_cup_2018 | 69 | 1.0867 | 1.2987 | 1.2041 | 1.6964 | 0.5217 | 1.0508 | 0.2415 |
| baseline_home_away | world_cup_2018 | 69 | 1.0598 | 1.2892 | 1.1905 | 1.5398 | 0.3623 | 1.0381 | 0.2358 |
| baseline_form | world_cup_2018 | 69 | 0.9996 | 1.2392 | 1.1111 | 1.6902 | 0.4928 | 1.0550 | 0.2414 |
| baseline_league_average | world_cup_2022 | 75 | 1.0733 | 1.3551 | 1.2681 | 1.6964 | 0.4133 | 1.0357 | 0.2447 |
| baseline_home_away | world_cup_2022 | 75 | 1.0546 | 1.3322 | 1.2335 | 1.5314 | 0.4933 | 1.0267 | 0.2396 |
| baseline_form | world_cup_2022 | 75 | 0.9883 | 1.2963 | 1.2016 | 1.6643 | 0.5467 | 0.9667 | 0.2190 |
