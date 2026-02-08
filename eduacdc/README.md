# eduacdc Package

The `eduacdc` package provides an educational implementation of the Atmospheric Cluster Dynamics Code (ACDC) for atmospheric outgrowth simulations. It models molecular clustering, collision/evaporation kinetics, and external losses (coagulation sink, wall loss).

---

## YAML Configuration

System and equations are loaded from YAML files via `InputParser.get_system_from_yaml()` and `InputParser.get_equations_from_yaml()`. The config schema is as follows.

### Top-level keys

| Key | Required | Description |
|-----|----------|--------------|
| `molecules` | Yes | Molecular species definitions (symbol → properties) |
| `clusters` | Yes | Cluster specifications by charge type |
| `conditions` | No | Ambient conditions (temperature, RH) |
| `cluster_properties` | Yes* | Thermodynamic data (*can be inferred from conditions) |
| `equations_configuration` | Yes | Outgrowth rules and equation options |
| `simulation` | No | Initial conditions and run settings (used by example scripts) |
| `process_coefficients_configuration` | No | Ion collision method, evaporations |

### molecules

Map of symbol (e.g. `A`, `N`, `B`) to properties:

- `name` (str), `charge` (int), `mass` (str/number, e.g. `"98.08 g/mol"`), `density` (str/number, e.g. `"1830 kg/m^3"`)
- `base_strength`, `acid_strength` (int, optional)
- For ions: `corresponding_neutral`, `corresponding_negative_ion`, `corresponding_positive_ion` (str, optional)

### clusters.specifications

Nested under `neutral`, `positive`, `negative`. Each entry is a dict mapping molecule symbol to:

- `[min, max]` (inclusive range)
- `[n]` (exact count)
- `n` (integer, exact count)

Example: `A: [1, 4]` and `N: [1, 4]` generates all (A,N) clusters with 1–4 of each.

Options: `include_generic_neg`, `include_generic_pos` (bool).

### conditions

- `temperature` (str, e.g. `"298.15 K"`)
- `relative_humidity` (float, 0–1)

### cluster_properties

- `reference_temperature`, `reference_pressure` (required)
- `energies`: map of cluster label (e.g. `2A`, `1A1N`) to:
  - Gibbs: single number or string (kcal/mol)
  - Enthalpy + entropy: `[enthalpy_kcal/mol, entropy_cal/(mol*K)]`
- Optional: `dipole_moments`, `polarizabilities`, `water_symbol`, `disable_hydration_parsing`

### equations_configuration

- `outgrowth_rules`: map of `neutral`/`positive`/`negative` to list of rules. Each rule is `{symbol: min_count}`; clusters matching any rule outgrow.
- `options`: `disable_nonmonomers`, `keep_useless_collisions`, `use_acid_base_logic`, `disable_evaporations`

### simulation (convention for example scripts)

- `initial_conditions`: map of cluster label to `{type, value}`:
  - `type`: `"constant"` (fixed conc), `"source"` (rate), `"initial"` (starting conc), `"sum_constant"` (with `independent_clusters`)
  - `value`: e.g. `"1e6 cm^-3"`, `"1 ppt"`, `"3 cm^-3/s"`
- `wall_loss_calculator`, `dilution_loss`, `run_method` (used by scripts, not parsed by core)

---

## eduacdc.core

Core domain models and physics for cluster dynamics.

| Module | Key Classes/Functions | Purpose |
|--------|------------------------|---------|
| **`molecules`** | `Molecule`, `MoleculeCollection`, `MoleculeType` | Molecular species definitions (name, symbol, charge, mass, density); supports neutral, ions, proton, missing proton |
| **`clusters`** | `Cluster`, `ClusterCollection`, `ClusterType`, `GenericIon` | Cluster definitions with composition, charge, mass; collections with indexing and filtering |
| **`cluster_properties`** | `ClusterProperties`, `EnergyData`, `compute_hydration_distribution` | Gibbs free energy, enthalpy, entropy; radius/diameter; hydrate distribution |
| **`process_coefficients`** | `ProcessCoefficients`, `ProcessCoefficientsConfiguration` | Collision and evaporation coefficients; Su82/constant ion methods |
| **`equations`** | `ClusterEquations`, `EquationsConfiguration`, `ProcessTracker` | SymPy-based ODE generation; collision/evaporation/boundary processes |
| **`system`** | `SimulationSystem`, `AmbientConditions` | System assembly (molecules, clusters, conditions); temperature, RH, saturation vapor pressure |
| **`coagulation_loss`** | `CoagulationLossCalculator`, `ExponentialCoagulationLoss`, `BackgroundCoagulationLoss`, `ConstantCoagulationLoss` | Size-dependent coagulation sink parameterizations |
| **`wall_loss`** | `WallLossCalculator`, `DiffusionWallLoss`, `Cloud4JAWallLoss`, `Cloud4SimpleWallLoss`, `ConstantWallLoss`, `ExternalWallLoss` | Flow-tube and chamber wall loss models |

---

## eduacdc.analysis

Analysis and visualization of ACDC systems and simulation results.

| Module | Key Functions | Purpose |
|--------|---------------|---------|
| **`fluxes`** | `create_flux_graph`, `plot_flux_graph`, `plot_flux_tracking`, `plot_final_outflux`, `plot_net_fluxes_heatmap`, `get_significant_outflux_reactions` | Flux network graphs, outflux identification, heatmaps |
| **`rates_and_deltags`** | `plot_reference_deltag_surface`, `plot_act_deltag_surface`, `plot_overall_evaporation_rate_surface`, `plot_monomer_collision_rate_ratio` | DeltaG surfaces, evaporation and collision rate plots |
| **`visualization`** | `plot_concentrations`, `plot_total_concentrations`, `plot_cluster_size_distribution`, `plot_formation_rates`, `plot_final_concentrations` | Time series, bar charts, size distributions |

---

## eduacdc.io

Input/output for ACDC configuration files.

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| **`parser`** | `InputParser` | `get_system_from_yaml(path)` → `SimulationSystem`; `get_equations_from_yaml(path, clusters)` → `ClusterEquations` |

---

## eduacdc.simulation

Simulation execution and result storage.

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| **`solver`** | `Simulation` | ODE integration (`scipy.integrate`); parses initial conditions (constant/source/initial); applies coagulation and wall loss |
| **`results`** | `SimulationResults` | Stores time, concentrations, coefficients, formation rates, outflux; provides `get_final_concentrations()`, total by charge type |

---

## eduacdc.utils

Utility functions and constants.

| Module | Key Exports | Purpose |
|--------|-------------|---------|
| **`constants`** | `ureg`, `BOLTZMANN_CONSTANT`, `PI`, `parse_quantity`, `format_quantity`, `conc` | Pint registry, physical constants, unit parsing, concentration context (ppt/ppb/ppm) |
