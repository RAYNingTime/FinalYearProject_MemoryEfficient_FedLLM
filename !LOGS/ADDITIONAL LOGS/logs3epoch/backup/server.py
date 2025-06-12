import flwr as fl
import gc
from flwr.server.strategy import FedAvgM


class MemoryEfficientFedAvgM(FedAvgM):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}
        # Use default aggregation from FedAvg
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        # Explicit memory cleanup
        del results
        del failures
        gc.collect()
        return aggregated_result



# Strategy with memory-efficient aggregation
strategy = MemoryEfficientFedAvgM(
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      min_fit_clients=2,
      min_evaluate_clients=3,
      min_available_clients=3,
      on_fit_config_fn=lambda rnd: {"epoch": rnd},
)


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
