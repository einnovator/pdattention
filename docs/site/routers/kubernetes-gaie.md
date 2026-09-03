# Kubernetes GAIE and llm-d

For elastic Kubernetes fleets, PRA compiles global eligibility and policy into cluster-local routing resources. It never chooses a pod on the request path.

| PRA object | Kubernetes object |
|---|---|
| Route | Gateway API `HTTPRoute` |
| ModelPool | GAIE `InferencePool` |
| BackendEndpoint | Pod state owned by Kubernetes discovery |
| Replica policy | Endpoint picker / llm-d scheduler |

The adapter currently emits a Kubernetes `v1/List` containing `inference.networking.k8s.io/v1` `InferencePool` resources and `gateway.networking.k8s.io/v1` `HTTPRoute` resources. Pool selectors include PRA model identity and optional qualification/profile/mode labels. The Endpoint Picker remains responsible for queue, load, cache locality, and readiness.

```yaml
router:
  id: gaie-eu
  kind: kubernetes-gaie
  management_url: https://pra-kubernetes-controller.internal
  region: eu
  cluster: prod-eu
  metadata:
    namespace: inference
    gateway: public-inference
```

```bash
pra router reconcile gaie-eu --confirm
kubectl apply -f /var/lib/pra/gaie-resources.yaml
```

Cluster-native discovery owns pods, HPA/KEDA/KServe own autoscaling, and the inference scheduler owns replicas. PRA only controls pool eligibility and route intent. See the [official GAIE v1 API](https://gateway-api-inference-extension.sigs.k8s.io/reference/spec/).

GitOps and local lab deployments may instead register an absolute generated
manifest path as a file URL, then apply that manifest with `kubectl` as shown.
