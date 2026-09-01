# Publishing a Community PRA Bundle

Community bundles use the same schema and integrity checks as project releases:

```bash
pra model inspect ORG/MODEL
pra model adapt ORG/MODEL -o .pra/adapters/model
pra adapter train FEATURES -o .pra/adapters/router
pra profiles calibrate ORG/MODEL -o .pra/runs/model
pra bundle build .pra/runs/model -o .pra/bundles/model
pra bundle validate .pra/bundles/model
pra bundle card .pra/bundles/model --update
pra hf push .pra/bundles/model USERNAME/pra-model --dry-run
pra hf push .pra/bundles/model USERNAME/pra-model --yes
```

The model card should state the exact base revision, training data, seeds,
adapter parameter count, measured engines and hardware, cohort size, evidence
tier, limitations, scripts, and source commit. Unknown values are
`NOT_MEASURED`, never zero.

Community releases retain the `community` trust label. Users can load them
explicitly with `-a USERNAME/REPO`; automatic resolution never selects an
untrusted repository. A later project qualification can add an immutable
revision to the trusted registry without changing the original release history.
