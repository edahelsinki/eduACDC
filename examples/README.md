# Example Configurations

YAML system configuration files for eduACDC simulations.

| File | Description |
|------|-------------|
| **ADW.yaml** | Sulfuric acid + dimethylamine (neutral only, with water) |
| **ANW.yaml** | Sulfuric acid + ammonia (neutral only, with water) |
| **AN_neutral_neg_pos.yaml** | Sulfuric acid + ammonia with neutral, negative, and positive ions |
| **three_component_system.yaml** | Sulfuric acid + ammonia + dimethylamine with ions |

Run a simulation using any of these configs:

```bash
python examples/run_simulation.py
```

Edit `run_simulation.py` to change the config path (default is `ANW.yaml`).

For the YAML configuration format and schema, see [eduacdc/README.md](../eduacdc/README.md#yaml-configuration).
