# Changes from the supplied prototype

| Prototype behavior | Repository behavior |
| --- | --- |
| Same target day's last/high/low/range/volatility enter features | All observed-price summaries shift by one calendar day |
| Feature quantiles computed on the full dataset | Whole-sample clipping removed |
| Log-price volatility fails on negative electricity prices | Within-day additive price-change standard deviation |
| Datetime inference can accept a numeric identifier | Explicit schema and unambiguous timestamp aliases |
| Price chosen by largest standard deviation | Explicit price selection or safe unambiguous aliases |
| Partial days treated as valid daily means | Complete declared UTC grid required |
| Median interval used for every OU transition | Actual elapsed time and exclusion of gap-spanning pairs |
| Nonpositive OU speed silently clamped | Invalid OU status and explicit ML fallback |
| Last sample assumed to coincide with midnight | Actual offset to midnight included in expectation |
| Successful GPU probe may have silently used CPU | Trained booster configuration verifies CUDA |
| Global warning suppression | Warnings visible; only probe warnings captured locally |
| Optional plotting imported unconditionally | Plotting dependency imported only when requested |
| Always-labelled `yhat_xgb` | Backend-neutral `yhat_ml` plus actual backend/device columns |
| MAPE near zero and missing-OU metric failures | MAE, RMSE, bias and explicit finite-pair counts |
| Machine-specific absolute paths | Portable CLI, Python API and output directories |
| Historical evaluation only | Separate backtest and unseen next-day forecast commands |
| No baseline or provenance | Two naive baselines, historical-level OU and run manifest |

The original conditioned-level hybrid idea and default 50/50 weight remain. Revised methodology intentionally changes the numerical results. The original script is not bundled: it contained private machine paths and its methodological issues are documented here instead.
