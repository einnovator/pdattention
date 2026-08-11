"""Train a compact router from precomputed frozen-backbone feature files."""

from pra_hf.training import load_feature_rows, train_router


train = load_feature_rows(["router_features_train.pt"])
validation = load_feature_rows(["router_features_validation.pt"])
router, metrics = train_router(train, validation, routing_width=128, device="cuda")
router.save_pretrained("my-pra-router")
print(metrics["validation"]["summary"])
