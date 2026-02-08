# eduACDC - Educational Atmospheric Cluster Dynamics Code

An educational Python implementation of the Atmospheric Cluster Dynamics Code (ACDC) for atmospheric outgrowth simulations.

## Quick Start

1. **Install:**

   ```bash
   pip install -e .   # or: pip install eduacdc (when published)
   ```

2. **See Examples:**
   - Explore the `examples/` directory for YAML system configuration files.
   - Use these configs with `eduacdc.io.InputParser` and `eduacdc.simulation.Simulation` to run outgrowth simulations.

## Code Layout

```mermaid
      graph TD
      subgraph MC[Molecules and Clusters]
         direction LR
         M[Molecules] --> C[Clusters]
      end
      subgraph RC[Rate Coefficients]
         ProcessCoeffConfig([Process Coefficients<br>Config]) -->Sys[[SimulationSystem]]
         SysCond([Ambient<br>Conditions]) --> Sys
         CP[Cluster Properties] --> Sys
      end
         MC --> Sys

      subgraph D[Differential Equations]
            EqConfig[Equations<br>Config] --> E
      end
      MC --> E[[ClusterEquations]]
      E --> Sim[[Simulation]]

      subgraph EL[External Losses]
         CL([Coagulation<br>Sink])
         WL([Wall])
         DL([Dilution])
      end

      Sim[[Simulation]]
      I[Initial<br>Conditions] --> Sim
      EL --> Sim
      MC ~~~ D
      MC ~~~ EL
      Sys --> Sim
      subgraph Results
         direction TB
         t[Time Steps]
         c[Cluster<br>Concentrations]
         f[Formation<br>Rates]
         flux[Fluxes<br>Rates]

      end
      Sim --> Results

```

## Project Structure

- `eduacdc/` — Main package code
- `examples/` — YAML system configuration files (ADW, ANW, AN_neutral_neg_pos, three_component_system)
