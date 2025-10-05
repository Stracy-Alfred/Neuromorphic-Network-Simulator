"""
Neuromorphic Network Simulator (Event-Driven, LIF + STDP)

- Core neuron model: Leaky Integrate-and-Fire (LIF) [implements leak and threshold dynamics]
  Vm(t) = Vrest + (Vm(t0) - Vrest) * exp(-(t - t0) / tau_m)  for t >= t0
  When Vm reaches Vth, the neuron emits a spike at that time, Vm -> Vreset and enters refractory.

- Synapse: connects pre -> post with weight W (interpreted as instantaneous delta-V on arrival)

- STDP (pair-based, nearest-neighbor):
  Let Δt = t_post - t_pre
   - If pre before post (Δt > 0): potentiation
       ΔW = A_plus * exp(-Δt / tau_plus)
   - If post before pre (Δt < 0): depression
       ΔW = -A_minus * exp(Δt / tau_minus)  [note Δt < 0 ⇒ magnitude uses |Δt|]

Implemented via two local updates:
  - On pre-spike at time t_pre: if last_post exists
        ΔW_pre = -A_minus * exp(-(t_pre - t_last_post) / tau_minus)
  - On post-spike at time t_post: if last_pre exists
        ΔW_post =  A_plus  * exp(-(t_post - t_last_pre)  / tau_plus)
Weights are clamped to [w_min, w_max].

Event-driven simulation:
  - Priority queue of events: (time, seq, kind, payload)
  - Kinds: 'external_spike', 'deliver'
  - No fixed time step; membrane potential is updated analytically at event times.

Demo network (main):
  - Two input neurons (as spike sources via external events) → One LIF output neuron
  - Schedule input0 spike so post spikes soon after, causing LTP on synapse0
  - Schedule input1 spike after the post spike, causing LTD on synapse1

Outputs:
  - Printed spike event log (time, neuron id)
  - Printed initial/final synaptic weights
  - Plot of Vm for the output neuron over time (saved to snn_output.png)
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Neuron:
    """Leaky Integrate-and-Fire (LIF) neuron.

    Units convention (for readability; treated as floats):
      - Voltages in millivolts (mV)
      - Time in seconds (s)

    Dynamics:
      Vm(t) = Vrest + (Vm_last - Vrest) * exp(-(t - t_last) / tau_m)
      When Vm >= Vth at an event time, emit spike, reset Vm to Vreset, enter refractory.
    """

    def __init__(
        self,
        neuron_id: int,
        name: str,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_th: float = -50.0,
        tau_m: float = 20e-3,  # 20 ms
        refractory_period: float = 5e-3,  # 5 ms
    ) -> None:
        self.id: int = neuron_id
        self.name: str = name
        self.v_rest: float = v_rest
        self.v_reset: float = v_reset
        self.v_th: float = v_th
        self.tau_m: float = tau_m
        self.refractory_period: float = refractory_period

        self.v_m: float = v_rest
        self.last_update_time: float = 0.0
        self.last_spike_time: Optional[float] = None

        self.incoming_synapses: List["Synapse"] = []
        self.outgoing_synapses: List["Synapse"] = []

        # Trace for plotting membrane potential over time
        self.trace_times: List[float] = [0.0]
        self.trace_vm: List[float] = [self.v_m]

    def _is_refractory(self, t: float) -> bool:
        if self.last_spike_time is None:
            return False
        return (t - self.last_spike_time) < self.refractory_period

    def decay_to(self, t: float) -> None:
        """Analytically decay Vm to time t, unless refractory holds Vm at Vreset.
        Updates last_update_time and records trace.
        """
        if t <= self.last_update_time:
            return
        if self._is_refractory(t):
            # During refractory, Vm stays at Vreset. Do not decay.
            self.v_m = self.v_reset
            self.last_update_time = t
            self._record(t)
            return
        # Exponential decay towards Vrest
        delta_t = t - self.last_update_time
        decay_factor = math.exp(-delta_t / self.tau_m)
        self.v_m = self.v_rest + (self.v_m - self.v_rest) * decay_factor
        self.last_update_time = t
        self._record(t)

    def receive_delta_v(self, t: float, delta_v: float, network: "Network") -> None:
        """Apply an instantaneous delta-V at time t (e.g., synaptic arrival).
        If Vm crosses threshold at t, emit a spike immediately.
        """
        # Update membrane to event time
        self.decay_to(t)
        if self._is_refractory(t):
            # Input has no effect during refractory
            return
        # Apply delta-V
        self.v_m += delta_v
        self._record(t)
        # Threshold check
        if self.v_m >= self.v_th:
            # Emit spike immediately at time t
            network._neuron_spike(self, t)

    def fire_external(self, t: float, network: "Network") -> None:
        """Force a spike at time t (used to model an external input source neuron).
        Outgoing synapses are notified; membrane is reset and refractory enforced.
        """
        # Before spiking, decay to t for consistency
        self.decay_to(t)
        network._neuron_spike(self, t)

    def _after_spike_reset(self, t: float) -> None:
        self.v_m = self.v_reset
        self.last_spike_time = t
        self.last_update_time = t
        self._record(t)

    def _record(self, t: float) -> None:
        if self.trace_times and t == self.trace_times[-1]:
            # Replace last value if two updates happen at same time
            self.trace_vm[-1] = self.v_m
        else:
            self.trace_times.append(t)
            self.trace_vm.append(self.v_m)


@dataclass
class Synapse:
    """Static synapse with event-driven delivery and pair-based STDP.

    weight is interpreted as an instantaneous delta-V (mV) applied to post Vm on arrival.
    """

    id: int
    pre: Neuron
    post: Neuron
    weight: float
    delay: float = 5e-3  # 5 ms conduction delay

    # STDP parameters
    A_plus: float = 0.8  # mV increase scale for LTP
    A_minus: float = 0.8  # mV decrease scale for LTD
    tau_plus: float = 20e-3
    tau_minus: float = 20e-3
    w_min: float = 0.0
    w_max: float = 50.0

    # Internal state for nearest-neighbor STDP
    last_pre_spike_time: Optional[float] = None
    last_post_spike_time: Optional[float] = None

    # Weight history for visualization / debugging
    history_times: List[float] = None
    history_weights: List[float] = None

    def __post_init__(self) -> None:
        self.history_times = [0.0]
        self.history_weights = [self.weight]

    def on_pre_spike(self, t_pre: float) -> None:
        """Apply LTD component if there was a recent post spike, then record t_pre.
        ΔW_pre = -A_minus * exp(-(t_pre - t_last_post) / tau_minus)
        """
        if self.last_post_spike_time is not None:
            delta_t = t_pre - self.last_post_spike_time
            if delta_t >= 0.0:
                delta_w = -self.A_minus * math.exp(-delta_t / self.tau_minus)
                self._update_weight(t_pre, delta_w)
        self.last_pre_spike_time = t_pre

    def on_post_spike(self, t_post: float) -> None:
        """Apply LTP component if there was a recent pre spike, then record t_post.
        ΔW_post = A_plus * exp(-(t_post - t_last_pre) / tau_plus)
        """
        if self.last_pre_spike_time is not None:
            delta_t = t_post - self.last_pre_spike_time
            if delta_t >= 0.0:
                delta_w = self.A_plus * math.exp(-delta_t / self.tau_plus)
                self._update_weight(t_post, delta_w)
        self.last_post_spike_time = t_post

    def _update_weight(self, t: float, delta_w: float) -> None:
        self.weight = max(self.w_min, min(self.w_max, self.weight + delta_w))
        self.history_times.append(t)
        self.history_weights.append(self.weight)

    def deliver(self, t: float, network: "Network") -> None:
        """Deliver synaptic delta-V to the post-synaptic neuron at time t."""
        self.post.receive_delta_v(t, self.weight, network)


class Network:
    """Event-driven SNN network with a priority queue of events."""

    def __init__(self) -> None:
        # Entities
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Synapse] = {}

        # Event queue: (time, seq, kind, payload)
        self._event_heap: List[Tuple[float, int, str, Dict[str, Any]]] = []
        self._seq_counter: int = 0
        self.current_time: float = 0.0

        # Logs
        self.spike_log: List[Tuple[float, int, str]] = []  # (t, neuron_id, neuron_name)

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons[neuron.id] = neuron

    def add_synapse(self, synapse: Synapse) -> None:
        self.synapses[synapse.id] = synapse
        synapse.pre.outgoing_synapses.append(synapse)
        synapse.post.incoming_synapses.append(synapse)

    # --- Event scheduling helpers ---
    def _push_event(self, t: float, kind: str, payload: Dict[str, Any]) -> None:
        self._seq_counter += 1
        heapq.heappush(self._event_heap, (t, self._seq_counter, kind, payload))

    def schedule_external_spike(self, neuron_id: int, t: float) -> None:
        self._push_event(t, "external_spike", {"neuron_id": neuron_id})

    def schedule_delivery(self, synapse_id: int, t: float) -> None:
        self._push_event(t, "deliver", {"synapse_id": synapse_id})

    # --- Core run loop ---
    def run(self, t_end: float) -> None:
        while self._event_heap and self._event_heap[0][0] <= t_end:
            t, _, kind, payload = heapq.heappop(self._event_heap)
            self.current_time = t
            if kind == "external_spike":
                neuron = self.neurons[payload["neuron_id"]]
                neuron.fire_external(t, self)
            elif kind == "deliver":
                syn = self.synapses[payload["synapse_id"]]
                syn.deliver(t, self)
            else:
                raise ValueError(f"Unknown event kind: {kind}")
        # Decay all neurons to t_end for final traces
        for neuron in self.neurons.values():
            neuron.decay_to(t_end)

    # --- Spike handling ---
    def _neuron_spike(self, neuron: Neuron, t: float) -> None:
        # Record spike
        self.spike_log.append((t, neuron.id, neuron.name))
        # Apply post-spike STDP on all incoming synapses
        for syn in neuron.incoming_synapses:
            syn.on_post_spike(t)
        # Notify outgoing synapses: apply pre-spike STDP and schedule delivery
        for syn in neuron.outgoing_synapses:
            syn.on_pre_spike(t)
            self.schedule_delivery(syn.id, t + syn.delay)
        # Reset neuron
        neuron._after_spike_reset(t)


# ------------------------- Demo and plotting -------------------------

def build_demo_network() -> Tuple[Network, Neuron, Neuron, Neuron, Synapse, Synapse]:
    net = Network()

    # Neurons: two inputs (0, 1), one output (2)
    n0 = Neuron(0, "Input0")
    n1 = Neuron(1, "Input1")
    n2 = Neuron(2, "Output", v_th=-50.0, v_rest=-65.0, v_reset=-65.0, tau_m=20e-3, refractory_period=5e-3)

    net.add_neuron(n0)
    net.add_neuron(n1)
    net.add_neuron(n2)

    # Synapses with initial weights (mV). Delay = 5 ms
    # s0 strong enough to drive the output to spike on its own
    s0 = Synapse(id=0, pre=n0, post=n2, weight=20.0, delay=5e-3)
    # s1 more modest; will be depressed by post-before-pre timing in demo
    s1 = Synapse(id=1, pre=n1, post=n2, weight=8.0, delay=5e-3)

    net.add_synapse(s0)
    net.add_synapse(s1)

    return net, n0, n1, n2, s0, s1


def run_demo() -> None:
    net, n0, n1, n2, s0, s1 = build_demo_network()

    sim_end = 0.2  # 200 ms

    # Schedule spikes:
    # - Input0 at 100 ms → arrives at 105 ms → Output spikes → LTP on s0
    # - Input1 at 106 ms → arrives at 111 ms after Output spike → LTD on s1
    net.schedule_external_spike(neuron_id=n0.id, t=0.100)
    net.schedule_external_spike(neuron_id=n1.id, t=0.106)

    # Capture initial weights
    initial_weights = {0: s0.weight, 1: s1.weight}

    # Run
    net.run(sim_end)

    # Results
    print("\n=== Event-Driven Spike Log ===")
    for t, nid, name in sorted(net.spike_log, key=lambda x: x[0]):
        print(f"t={t*1e3:7.2f} ms | Neuron {nid} ({name}) spiked")

    print("\n=== Weight Evolution (initial → final) ===")
    print(f"Synapse 0 (Input0→Output): {initial_weights[0]:.3f} mV → {s0.weight:.3f} mV")
    print(f"Synapse 1 (Input1→Output): {initial_weights[1]:.3f} mV → {s1.weight:.3f} mV")

    # Plot membrane potential of Output neuron
    plt.figure(figsize=(8, 4))
    times_ms = [t * 1e3 for t in n2.trace_times]
    plt.plot(times_ms, n2.trace_vm, label="Output Vm (mV)")

    # Mark spike times
    output_spike_times = [t for (t, nid, _) in net.spike_log if nid == n2.id]
    for t in output_spike_times:
        plt.axvline(t * 1e3, color="r", linestyle="--", alpha=0.5)

    plt.title("Output Neuron Membrane Potential (Event-Driven LIF)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm (mV)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_output.png", dpi=150)
    print("\nSaved membrane potential plot to 'snn_output.png'.")


if __name__ == "__main__":
    run_demo()
