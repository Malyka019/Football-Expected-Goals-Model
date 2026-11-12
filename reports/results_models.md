# Step 4 - model comparison

Poisson GLM uses the engineered features; Dixon-Coles fits per-team attack/defence with two-year time decay and the rho low-score correction. All models are fitted on the train block only.

| model | split | n_matches | mae | rmse | poisson_deviance | mean_predicted_goals | accuracy | log_loss | rps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_league_average | validation | 3562 | 1.1910 | 1.5763 | 1.7053 | 1.4906 | 0.2914 | 1.0741 | 0.2367 |
| baseline_home_away | validation | 3562 | 1.1583 | 1.5578 | 1.6644 | 1.4806 | 0.4773 | 1.0468 | 0.2263 |
| baseline_form | validation | 3562 | 1.0528 | 1.4438 | 1.4490 | 1.3585 | 0.5255 | 0.9795 | 0.2040 |
| poisson_glm | validation | 3562 | 0.9577 | 1.3038 | 1.2234 | 1.4098 | 0.6033 | 0.8611 | 0.1673 |
| dixon_coles | validation | 3562 | 0.9471 | 1.2895 | 1.2085 | 1.3477 | 0.6047 | 0.8660 | 0.1687 |
| gbm_poisson | validation | 3562 | 0.9528 | 1.2947 | 1.2043 | 1.4204 | 0.6002 | 0.8622 | 0.1678 |
| baseline_league_average | test | 970 | 1.0769 | 1.3363 | 1.4652 | 1.4906 | 0.2753 | 1.0721 | 0.2372 |
| baseline_home_away | test | 970 | 1.0275 | 1.3102 | 1.4146 | 1.4560 | 0.4979 | 1.0416 | 0.2258 |
| baseline_form | test | 970 | 0.9640 | 1.2617 | 1.3098 | 1.3311 | 0.5052 | 1.0158 | 0.2150 |
| poisson_glm | test | 970 | 0.8973 | 1.1799 | 1.1609 | 1.3531 | 0.5794 | 0.9160 | 0.1835 |
| dixon_coles | test | 970 | 0.8979 | 1.1841 | 1.1752 | 1.2750 | 0.5742 | 0.9340 | 0.1890 |
| gbm_poisson | test | 970 | 0.9068 | 1.1755 | 1.1526 | 1.3824 | 0.5794 | 0.9086 | 0.1814 |
| baseline_league_average | world_cup_2018 | 64 | 0.9432 | 1.1645 | 1.1523 | 1.4906 | 0.4062 | 1.0617 | 0.2401 |
| baseline_home_away | world_cup_2018 | 64 | 0.8983 | 1.1411 | 1.1155 | 1.3225 | 0.3906 | 1.0666 | 0.2400 |
| baseline_form | world_cup_2018 | 64 | 0.8406 | 1.1009 | 1.0460 | 1.2648 | 0.5312 | 1.0019 | 0.2162 |
| poisson_glm | world_cup_2018 | 64 | 0.8505 | 1.1057 | 1.0626 | 1.2609 | 0.5312 | 1.0046 | 0.2168 |
| dixon_coles | world_cup_2018 | 64 | 0.8309 | 1.1013 | 1.0522 | 1.1162 | 0.5938 | 0.9739 | 0.2057 |
| gbm_poisson | world_cup_2018 | 64 | 0.8601 | 1.1123 | 1.0656 | 1.3106 | 0.5469 | 0.9903 | 0.2136 |
| baseline_league_average | world_cup_2022 | 64 | 1.0761 | 1.3452 | 1.4356 | 1.4906 | 0.3281 | 1.0754 | 0.2363 |
| baseline_home_away | world_cup_2022 | 64 | 1.0468 | 1.3485 | 1.4403 | 1.3146 | 0.4375 | 1.0908 | 0.2419 |
| baseline_form | world_cup_2022 | 64 | 0.9809 | 1.3110 | 1.3900 | 1.2527 | 0.4219 | 1.0753 | 0.2296 |
| poisson_glm | world_cup_2022 | 64 | 0.9865 | 1.3030 | 1.3334 | 1.2673 | 0.5000 | 1.0590 | 0.2235 |
| dixon_coles | world_cup_2022 | 64 | 0.9740 | 1.2961 | 1.3378 | 1.1284 | 0.5625 | 1.0064 | 0.2049 |
| gbm_poisson | world_cup_2022 | 64 | 0.9695 | 1.2829 | 1.3138 | 1.3160 | 0.4844 | 1.0518 | 0.2225 |
