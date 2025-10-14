"""
Neuromorphic Network Simulator - Event-Driven SNN in a Single File

Implements:
- Event-driven simulation engine using a priority queue (heapq)
- Leaky Integrate-and-Fire (LIF) neurons with analytical updates between events
- Synapses with nearest-neighbor, pair-based STDP (pre/post local updates)
- Simple demonstration: 2 inputs -> 1 output, showing LTP and LTD

All times are in milliseconds (ms). Voltages are in arbitrary units (a.u.).
Weights are unitless, applied as instantaneous EPSP jumps upon arrival.

This file is self-contained and runnable. See run_demo() at bottom.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ----------------------------- Utility Types ----------------------------- #

EventPayload = Dict[str, object]


# ----------------------------- Neuron Model ------------------------------ #

@dataclass
class Neuron:
    """Leaky Integrate-and-Fire neuron with analytical membrane updates.

    The membrane potential evolves between events as:
        V(t) = V_rest + (V(t0) - V_rest) * exp(-(t - t0) / tau_m)
    
    - When a spike occurs, V is reset to V_reset and held for refractory_period.
    - During refractory, V is clamped at V_reset (inputs are ignored for spiking),
      then normal integration resumes after refractory.
    """

    neuron_id: int
    name: str

    # LIF parameters
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau_m: float = 20.0  # membrane time constant (ms)
    refractory_period: float = 5.0  # ms

    # State
    v_mem: float = 0.0
    last_update_time: float = 0.0
    refractory_until: float = 0.0

    # Mark this neuron as a spike generator (input source). It will not be used
    # for threshold-based spiking; instead, spikes are injected externally.
    is_input_generator: bool = False

    # For logging / analysis
    last_spike_time: Optional[float] = None

    def in_refractory(self, t: float) -> bool:
        return t < self.refractory_until

    def update_potential_to(self, t: float) -> float:
        """Analytically update V_m to time t.

        Handles refractory clamping if it overlaps the interval.
        """
        if t <= self.last_update_time:
            return self.v_mem

        # If we are in refractory for some or all of [last_update_time, t]
        if self.last_update_time < self.refractory_until:
            # If the whole interval is within refractory
            if t <= self.refractory_until:
                self.v_mem = self.v_reset
                self.last_update_time = t
                return self.v_mem
            # Otherwise, clamp until refractory end, then integrate decay after
            self.v_mem = self.v_reset
            self.last_update_time = self.refractory_until

        # Free decay from last_update_time to t
        dt = t - self.last_update_time
        if dt > 0.0:
            decay = math.exp(-dt / self.tau_m)
            self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * decay
            self.last_update_time = t
        return self.v_mem

    def receive_excitation(self, t: float, delta_v: float) -> float:
        """Apply an instantaneous EPSP (voltage jump) at time t.

        This should be called only after update_potential_to(t).
        During refractory, we clamp to v_reset; the excitation does not trigger
        spiking but still can influence V after refractory if you choose to
        accumulate. For simplicity here, we keep V at v_reset during refractory.
        """
        if self.in_refractory(t):
            # Ignore voltage changes during refractory in this simple model.
            self.v_mem = self.v_reset
            return self.v_mem
        self.v_mem += float(delta_v)
        return self.v_mem

    def should_spike(self) -> bool:
        # Refractory is checked by the caller; here we only compare to threshold
        return self.v_mem >= self.v_threshold

    def enter_refractory(self, t: float) -> None:
        self.refractory_until = t + self.refractory_period
        self.last_spike_time = t
        self.v_mem = self.v_reset
        self.last_update_time = t


# ----------------------------- Synapse Model ----------------------------- #

@dataclass
class Synapse:
    """Static-delay synapse with pair-based nearest-neighbor STDP.

    STDP is applied locally:
    - On pre arrival at synapse (time t_pre_arrival): LTD based on last post spike
    - On post spike at soma (time t_post): LTP based on last pre arrival

    Multiplicative, hard-bounded weight updates:
      LTP: w += A_plus  * (w_max - w) * exp(-Delta_t / tau_plus)   for Delta_t >= 0
      LTD: w -= A_minus * (w - w_min) * exp(-Delta_t / tau_minus)  for Delta_t >= 0

    where Delta_t = t_post - t_pre_arrival (for LTP) and
          Delta_t = t_pre_arrival - t_post (for LTD).
    """

    synapse_id: int
    label: str
    pre_id: int
    post_id: int

    weight: float
    delay: float  # ms, axonal delay from pre soma spike to arrival at synapse

    # STDP parameters
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.05
    a_minus: float = 0.05
    w_min: float = 0.0
    w_max: float = 2.0

    # Local spike times (nearest-neighbor)
    last_pre_arrival_time: Optional[float] = None
    last_post_spike_time: Optional[float] = None

    def update_on_pre_arrival(self, t_pre_arrival: float) -> None:
        """Apply LTD based on the most recent post spike time at this synapse."""
        if self.last_post_spike_time is None:
            self.last_pre_arrival_time = t_pre_arrival
            return
        delta = t_pre_arrival - self.last_post_spike_time
        if delta >= 0.0:
            depression = self.a_minus * (self.weight - self.w_min) * math.exp(-delta / self.tau_minus)
            self.weight = max(self.w_min, self.weight - depression)
        self.last_pre_arrival_time = t_pre_arrival

    def update_on_post_spike(self, t_post: float) -> None:
        """Apply LTP based on the most recent pre arrival time at this synapse."""
        if self.last_pre_arrival_time is None:
            self.last_post_spike_time = t_post
            return
        delta = t_post - self.last_pre_arrival_time
        if delta >= 0.0:
            potentiation = self.a_plus * (self.w_max - self.weight) * math.exp(-delta / self.tau_plus)
            self.weight = min(self.w_max, self.weight + potentiation)
        self.last_post_spike_time = t_post


# --------------------------- Network and Engine -------------------------- #

class Network:
    """Event-driven spiking neural network using a priority queue scheduler."""

    def __init__(self) -> None:
        self.time: float = 0.0
        self._event_queue: List[Tuple[float, int, str, EventPayload]] = []
        self._event_seq: int = 0

        self.neurons_by_id: Dict[int, Neuron] = {}
        self.neuron_name_to_id: Dict[str, int] = {}

        self.synapses_by_id: Dict[int, Synapse] = {}
        self.outgoing_by_pre: Dict[int, List[int]] = {}
        self.incoming_by_post: Dict[int, List[int]] = {}

        # Spike log for demonstration
        self.spike_log: List[Tuple[float, str]] = []  # (time, neuron_name)

        # For plotting membrane potential of a chosen neuron (e.g., output)
        self.tracked_neuron_id: Optional[int] = None
        self.tracked_vm_trace: List[Tuple[float, float]] = []  # (time, Vm)

    # ------------------------- Construction API ------------------------- #

    def add_neuron(self, name: str, **lif_params: float) -> int:
        neuron_id = len(self.neurons_by_id)
        neuron = Neuron(neuron_id=neuron_id, name=name, **lif_params)
        self.neurons_by_id[neuron_id] = neuron
        self.neuron_name_to_id[name] = neuron_id
        return neuron_id

    def connect(self, pre_id: int, post_id: int, weight: float, delay: float,
                label: Optional[str] = None, **stdp_params: float) -> int:
        synapse_id = len(self.synapses_by_id)
        syn = Synapse(
            synapse_id=synapse_id,
            label=label or f"S{synapse_id}",
            pre_id=pre_id,
            post_id=post_id,
            weight=weight,
            delay=delay,
            **stdp_params,
        )
        self.synapses_by_id[synapse_id] = syn
        self.outgoing_by_pre.setdefault(pre_id, []).append(synapse_id)
        self.incoming_by_post.setdefault(post_id, []).append(synapse_id)
        return synapse_id

    def track_neuron_vm(self, neuron_id: int) -> None:
        self.tracked_neuron_id = neuron_id
        self.tracked_vm_trace.clear()

    # --------------------------- Event Handling -------------------------- #

    def schedule_input_spike(self, neuron_id: int, t: float) -> None:
        self._push_event(t, "input_spike", {"neuron_id": neuron_id})

    def _push_event(self, t: float, kind: str, payload: EventPayload) -> None:
        self._event_seq += 1
        heapq.heappush(self._event_queue, (float(t), self._event_seq, kind, payload))

    def run(self, until: Optional[float] = None) -> None:
        while self._event_queue:
            t, _, kind, payload = heapq.heappop(self._event_queue)
            if until is not None and t > until:
                # Put back and stop
                heapq.heappush(self._event_queue, (t, self._event_seq, kind, payload))
                break
            self.time = t
            if kind == "input_spike":
                self._handle_input_spike(t, payload)
            elif kind == "syn_arrival":
                self._handle_syn_arrival(t, payload)
            elif kind == "end_refractory":
                self._handle_end_refractory(t, payload)
            else:
                raise ValueError(f"Unknown event kind: {kind}")

    # ------------------------------ Handlers ----------------------------- #

    def _handle_input_spike(self, t: float, payload: EventPayload) -> None:
        neuron_id = int(payload["neuron_id"])  # pre-soma spike
        neuron = self.neurons_by_id[neuron_id]
        # Log spike for input generator as well
        neuron.last_spike_time = t
        self.spike_log.append((t, neuron.name))
        # Deliver to outgoing synapses after delay
        for syn_id in self.outgoing_by_pre.get(neuron_id, []):
            syn = self.synapses_by_id[syn_id]
            self._push_event(t + syn.delay, "syn_arrival", {"synapse_id": syn_id})

    def _handle_syn_arrival(self, t: float, payload: EventPayload) -> None:
        synapse_id = int(payload["synapse_id"])  # pre arrival time at synapse
        syn = self.synapses_by_id[synapse_id]
        post = self.neurons_by_id[syn.post_id]

        # STDP: local pre-arrival update (LTD if post spiked earlier)
        syn.update_on_pre_arrival(t)

        # Update post membrane to current time and optionally track Vm
        v_before = post.update_potential_to(t)
        if self.tracked_neuron_id == post.neuron_id:
            # Record decayed value right before synaptic jump
            self._record_vm(t, v_before)

        # Apply instantaneous EPSP if not in refractory
        v_after_jump = post.receive_excitation(t, syn.weight)
        if self.tracked_neuron_id == post.neuron_id:
            # Record immediate jump at the same time
            self._record_vm(t, v_after_jump)

        # Check for spike after the jump
        if (not post.is_input_generator) and (not post.in_refractory(t)) and post.should_spike():
            self._emit_spike(post, t)

    def _handle_end_refractory(self, t: float, payload: EventPayload) -> None:
        neuron_id = int(payload["neuron_id"]) 
        neuron = self.neurons_by_id[neuron_id]
        # Bring neuron to t (clamped) to ensure logs are consistent
        v = neuron.update_potential_to(t)
        if self.tracked_neuron_id == neuron_id:
            self._record_vm(t, v)
        # Nothing else required; integration resumes after t

    # ---------------------------- Spike Emission -------------------------- #

    def _emit_spike(self, neuron: Neuron, t: float) -> None:
        # Log spike
        self.spike_log.append((t, neuron.name))

        # STDP: on post spike, update all incoming synapses to this neuron (LTP)
        for syn_id in self.incoming_by_post.get(neuron.neuron_id, []):
            syn = self.synapses_by_id[syn_id]
            syn.update_on_post_spike(t)

        # Enter refractory and schedule end event
        neuron.enter_refractory(t)
        if self.tracked_neuron_id == neuron.neuron_id:
            self._record_vm(t, neuron.v_mem)  # reset point
        self._push_event(t + neuron.refractory_period, "end_refractory", {"neuron_id": neuron.neuron_id})

        # Forward spike to downstream synapses
        for syn_id in self.outgoing_by_pre.get(neuron.neuron_id, []):
            syn = self.synapses_by_id[syn_id]
            self._push_event(t + syn.delay, "syn_arrival", {"synapse_id": syn_id})

    # ---------------------------- VM Tracking ----------------------------- #

    def _record_vm(self, t: float, v: float) -> None:
        # Avoid duplicating identical consecutive points
        if self.tracked_vm_trace and self.tracked_vm_trace[-1] == (t, v):
            return
        self.tracked_vm_trace.append((t, v))

    def render_tracked_vm_plot(self, filename: str, tail_ms: float = 20.0) -> None:
        """Generate a continuous-looking plot from event-time Vm samples.

        Between successive (t0, v0) -> (t1, v1_before_jump) points where the neuron
        was in free decay, the exact Vm(t) is known analytically, so we sample a
        dense curve using that exponential form. Vertical steps are inserted at
        event times to show EPSP jumps and resets.
        """
        if self.tracked_neuron_id is None or not self.tracked_vm_trace:
            return

        try:
            import matplotlib
        except Exception:
            # Attempt a best-effort runtime install for portability
            try:
                import sys, subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib"], check=False)
                import matplotlib  # retry
            except Exception as exc:
                print(f"[WARN] Could not import/install matplotlib for plotting: {exc}")
                return

        try:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[WARN] Matplotlib backend init failed: {exc}")
            return

        neuron = self.neurons_by_id[self.tracked_neuron_id]
        # Extend with a tail so decay to rest is visible
        last_t, last_v = self.tracked_vm_trace[-1]
        # Bring neuron state forward to last_t (already done), then to tail end
        t_end = last_t + tail_ms
        # Generate synthetic tail by decaying from the last known point
        num_tail_points = 40
        tail_times = [last_t + i * (tail_ms / num_tail_points) for i in range(1, num_tail_points + 1)]
        # For tail, use exponential decay from last_v towards v_rest
        tail_values = [
            neuron.v_rest + (last_v - neuron.v_rest) * math.exp(-(tt - last_t) / neuron.tau_m)
            for tt in tail_times
        ]

        # Build dense samples between recorded points
        dense_t: List[float] = []
        dense_v: List[float] = []
        for idx in range(len(self.tracked_vm_trace) - 1):
            (t0, v0) = self.tracked_vm_trace[idx]
            (t1, v1) = self.tracked_vm_trace[idx + 1]
            # If same time, draw a vertical step
            if abs(t1 - t0) < 1e-12:
                # vertical step: add both points
                if not dense_t:
                    dense_t.append(t0)
                    dense_v.append(v0)
                dense_t.append(t1)
                dense_v.append(v1)
                continue
            # Otherwise sample exponential decay from (t0, v0) to (t1, ~)
            num_seg_points = max(8, int((t1 - t0) / 1.0) * 8)
            for i in range(num_seg_points + 1):
                tt = t0 + (t1 - t0) * (i / num_seg_points)
                vv = neuron.v_rest + (v0 - neuron.v_rest) * math.exp(-(tt - t0) / neuron.tau_m)
                dense_t.append(tt)
                dense_v.append(vv)
        # Append the last recorded point and the tail
        last_t_rec, last_v_rec = self.tracked_vm_trace[-1]
        dense_t.append(last_t_rec)
        dense_v.append(last_v_rec)
        dense_t.extend(tail_times)
        dense_v.extend(tail_values)

        plt.figure(figsize=(8, 3))
        plt.plot(dense_t, dense_v, lw=2.0, color="#1f77b4")
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane potential V_m (a.u.)")
        plt.title(f"Membrane potential of {self.neurons_by_id[self.tracked_neuron_id].name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()


# ------------------------------ Demonstration ---------------------------- #

def run_demo() -> None:
    """Run the requested 2->1 demonstration with LTP and LTD cases.

    Setup:
      - N0, N1 are input spike generators
      - N_out is a LIF neuron receiving from N0 via S0, and from N1 via S1

    Plan:
      - Schedule N0 spike at 100 ms. With S0 delay=5 ms and strong weight, it
        arrives at 105 ms and triggers N_out to spike at 105 ms.
      - Schedule N1 spike at 115 ms. With S1 delay=5 ms and weaker weight, it
        arrives at 120 ms, well after N_out has already spiked, inducing LTD on S1.
    """
    net = Network()

    # Create neurons
    n0 = net.add_neuron("N0", is_input_generator=True)
    n1 = net.add_neuron("N1", is_input_generator=True)
    n_out = net.add_neuron(
        "N_out",
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_m=20.0,
        refractory_period=5.0,
        is_input_generator=False,
    )

    net.track_neuron_vm(n_out)

    # Connect synapses with same delay, different weights
    # S0: strong enough to trigger spiking upon arrival
    s0 = net.connect(n0, n_out, weight=1.10, delay=5.0, label="S0",
                     tau_plus=20.0, tau_minus=20.0, a_plus=0.05, a_minus=0.05,
                     w_min=0.0, w_max=2.0)
    # S1: weaker; will arrive after the N_out spike -> LTD expected
    s1 = net.connect(n1, n_out, weight=0.60, delay=5.0, label="S1",
                     tau_plus=20.0, tau_minus=20.0, a_plus=0.05, a_minus=0.05,
                     w_min=0.0, w_max=2.0)

    # Print initial weights
    print("Initial Weights:")
    print(f"  S0 (N0->N_out): {net.synapses_by_id[s0].weight:.4f}")
    print(f"  S1 (N1->N_out): {net.synapses_by_id[s1].weight:.4f}")

    # Schedule spikes per test cases
    net.schedule_input_spike(n0, 100.0)  # arrives at 105 ms -> triggers LTP on S0
    net.schedule_input_spike(n1, 115.0)  # arrives at 120 ms -> LTD on S1

    # Run simulation
    net.run()

    # Print spike log
    print("\nEvent-Driven Spike Log (time ms, neuron):")
    for t, name in sorted(net.spike_log, key=lambda x: x[0]):
        print(f"  {t:7.3f}  {name}")

    # Print final weights
    print("\nFinal Weights:")
    print(f"  S0 (N0->N_out): {net.synapses_by_id[s0].weight:.4f}")
    print(f"  S1 (N1->N_out): {net.synapses_by_id[s1].weight:.4f}")

    # Generate plot for output neuron's membrane potential
    plot_file = "snn_results.png"
    net.render_tracked_vm_plot(plot_file)
    print(f"\nSaved membrane potential plot to: {plot_file}")


if __name__ == "__main__":
    run_demo()
