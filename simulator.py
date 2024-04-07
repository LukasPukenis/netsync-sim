import random
import typing
from typing import List, TypedDict

import matplotlib.pyplot as plt
import numpy as np

TOTAL_SIMULATION_TIME = 20000
batch = True
RAND_SEED = 42

# TODO move outside to interactive sliders
MIN_SIGNAL_TS = 0
MAX_SIGNAL_TS = 1000

# TODO: constants above could go into the notebook


class RadioEventDirection:
    Incoming = 0
    Outgoing = 1

    def __repr__(self):
        return f"RadioEventDirection: {self.Incoming}, {self.Outgoing}"


class RadioEvent:
    def __init__(self, time: int, name: str, dir: int):
        self.time = time
        self.name = name
        self.dir = dir

    def __repr__(self):
        return f"RadioEvent: {self.time}, {self.name}, {self.dir}"


class RadioAggregator:
    def __init__(self):
        self.events = []
        self.states = []

    def add_radio_state(self, radio_state: int, time: int) -> None:
        self.states.append((time, radio_state))

    def add_event(self, event: RadioEvent) -> None:
        self.events.append(event)

    def __repr__(self):
        return f"RadioAggregator: {self.events} {self.states}"


class Time:
    def __init__(self):
        # non-zero to simplify the example as we don't need to check against zero
        self._curr_time = 0

    def current_time(self) -> int:
        return self._curr_time

    def advance_time(self) -> None:
        self._curr_time += 1

    def reset(self):
        self._curr_time = 0

    def __repr__(self):
        return f"Time: {self._curr_time}"


time = Time()


class EmitTime:
    def __init__(self, start: int, end: int):
        self.premature_time = start
        self.desired_time = end

        assert self.premature_time < self.desired_time

    def includes(self, time: int) -> bool:
        return self.premature_time <= time <= self.desired_time

    def __repr__(self):
        return f"EmitTime: {self.premature_time}, {self.desired_time}"


class IncomingSignal:
    def __init__(self, name: str, received_time_ts: int, src_node: "Device"):
        self.name = name
        self.received_time_ts = received_time_ts
        self._src_node = src_node

    def __repr__(self):
        return f"IncomingSignal: {self.name}, {self.received_time_ts}, {self._src_node}"


class Signal:
    def __init__(
        self,
        name: str,
        period: int,
        delta: int,
        dest_node=None,
        src_node=None,
        batch: bool = False,
    ):
        self.src_node = src_node
        self.name = name
        self.period = period
        self.delta = delta
        self.dest_node = dest_node
        self.last_emit_time: typing.Optional[int] = None
        self.batch = batch

        # signal might not be initiated immediately, so immitate some delay to misalign
        t = time.current_time() + random.randint(MIN_SIGNAL_TS, MAX_SIGNAL_TS)
        self.emit_zone = EmitTime(t - delta, t)

    def post_emit(self, time: int) -> None:
        new_time = time + self.period
        self.emit_zone = EmitTime(
            new_time - self.delta,
            new_time,
        )
        self.last_emit_time = time

    def can_emit_at(self, time: int) -> bool:
        return self.emit_zone.includes(time)

    def convert_to_incoming(self) -> IncomingSignal:
        assert self.src_node is not None
        return IncomingSignal(self.name, time.current_time(), self.src_node)

    def __repr__(self):
        return f"Signal: {self.name}, {self.period}, {self.delta}, {self.emit_zone} {self.dest_node}"


class Device:
    def __init__(
        self, name: str, local_batching: bool, recv_batching: bool, min_or_max: str
    ):
        self.name = name
        self.radio: typing.Optional[Radio] = None
        self.outgoing_signals: typing.Optional[list[Signal]] = None

        self._local_batching = local_batching
        self._recv_batching = recv_batching

        self._min_or_max = min_or_max
        self._peer_last_received = {}

    def add_radio(self, radio) -> None:
        assert self.radio is None

        self.radio = radio

    def add_outgoing_signals(self, signals: list[Signal]) -> None:
        self.outgoing_signals = signals

    def receive(self, signal: IncomingSignal) -> None:
        assert self.radio is not None

        self._peer_last_received[signal._src_node.name] = True

    def emit(self, signal: Signal) -> None:
        signal.post_emit(time.current_time())

        assert self.radio is not None
        self.radio.emit(signal)

        self._peer_last_received[signal.dest_node.name] = False

    def update(self) -> None:
        if self.outgoing_signals is None:
            return

        assert self.radio
        curr_time = time.current_time()

        if self._recv_batching:
            # when receive batching is enabled, it means this is a good opportunity
            # to emit data towards the peer from which data came
            # TODO: can't really get this one to work properly, it produces worse results than local batching for some reason
            # though by reasoning it should not. I tried having minimum cooldown period but that doesn't help either
            for signal in self.outgoing_signals:
                if (
                    signal.dest_node.name in self._peer_last_received
                    and self._peer_last_received[signal.dest_node.name]
                ):
                    for signal in self.outgoing_signals:
                        if (
                            signal.emit_zone.premature_time + (signal.period // 2)
                            <= curr_time
                            <= signal.emit_zone.desired_time
                        ):
                            self.emit(signal)

        if self._local_batching:
            signals_to_emit = []
            for signal in self.outgoing_signals:
                if signal.can_emit_at(curr_time):
                    signals_to_emit.append(signal)

            min_time = -9999999  # TODO HERE WAS A BUG!!!
            max_time = 999999

            # it's not worth emitting this event if the batch is small or else we lose all the beenfit
            # we can't really check if all signals align as that's unrealistic in case the periods are all not equal
            if len(signals_to_emit) > 1:
                # calculate best slice of time
                for signal in signals_to_emit:
                    min_time = max(min_time, signal.emit_zone.premature_time)
                    max_time = min(max_time, signal.emit_zone.desired_time)

                if self._min_or_max == "avg":
                    target_time = (min_time + max_time) // 2
                elif self._min_or_max == "min":
                    target_time = min_time
                elif self._min_or_max == "max":
                    target_time = max_time
                else:
                    raise ValueError(f"Unknown min_or_max: {self._min_or_max}")

                if curr_time == target_time:
                    for signal in signals_to_emit:
                        self.emit(signal)

        # make sure we emit on hard deadline of a signal
        # this is crucial to call at the very end to guarantee emission of events
        for signal in self.outgoing_signals:
            if signal.emit_zone.desired_time == curr_time:
                self.emit(signal)

    def __repr__(self):
        return f"Device: {self.name}"


class Network:  #
    def __init__(self, latency: tuple[int, int]):
        assert latency[0] < latency[1]

        self.latency = latency

        # ingress queue is used to simulate the sending of packets. packet is stored with timestamp and then sent to the deveice once the time arrives
        self.ingress_queue: typing.List[
            typing.Tuple[int, typing.Tuple[Device, IncomingSignal]]
        ] = []

    def set_latency(self, latency: tuple[int, int]) -> None:
        assert latency[0] > latency[1]

        self.latency = latency

    def send(self, dest_node: Device, signal: Signal) -> None:
        lat = np.random.randint(self.latency[0], self.latency[1] + 1)
        self.ingress_queue.append(
            (int(time.current_time() + lat), (dest_node, signal.convert_to_incoming()))
        )

    def update(self) -> None:
        for t, (dest_node, signal) in self.ingress_queue.copy():
            if t == time.current_time():
                assert dest_node.radio
                # TODO: can this be simulated nicer?
                dest_node.radio.receive(signal)
                self.ingress_queue.remove((t, (dest_node, signal)))


class RadioState:
    Low = 0
    High = 1

    def __repr__(self):
        return f"RadioState: {self.Low}, {self.High}"


class Radio:
    def __init__(
        self, node: Device, aggregator: RadioAggregator, network: Network, cooldown: int
    ):
        self.state = RadioState.Low
        self.network = network
        self._cooldown = cooldown
        self.state_change_countdown = cooldown
        self.node = node
        self.aggregator = aggregator
        self.aggregator.add_radio_state(self.state, time.current_time())

    def _set_to_high(self) -> None:
        self.state = RadioState.High
        self.aggregator.add_radio_state(self.state, time.current_time())
        self.state_change_countdown = self._cooldown

    def receive(self, signal: IncomingSignal) -> None:
        self._set_to_high()
        self.aggregator.add_event(
            RadioEvent(time.current_time(), signal.name, RadioEventDirection.Incoming)
        )
        self.node.receive(signal)

    def emit(self, signal: Signal) -> None:
        self._set_to_high()
        self.aggregator.add_event(
            RadioEvent(time.current_time(), signal.name, RadioEventDirection.Outgoing)
        )

        self.network.send(signal.dest_node, signal)

    def update(self) -> None:
        if self.state == RadioState.High:
            if self.state_change_countdown > 0:
                self.state_change_countdown -= 1
            else:
                self.state = RadioState.Low
                self.aggregator.add_radio_state(self.state, time.current_time())

    def __repr__(self):
        return f"Radio: {self.state}, {self.state_change_countdown}"


class StateInfo(TypedDict):
    time: int
    nodes: List[Device]
    radios: List[Radio]


class StateMachine:
    def __init__(self):
        self.network = None
        self.nodes = []
        self.radios = []

    def add_network(self, network: Network) -> None:
        self.network = network

    def add_node(self, node: Device) -> None:
        self.nodes.append(node)

    def add_radio(self, radio: Radio) -> None:
        self.radios.append(radio)

    def update(self) -> None:
        self.network.update()

        for node in self.nodes:
            node.update()

        for radio in self.radios:
            radio.update()

        time.advance_time()

    # return all of the state
    def state(self) -> StateInfo:
        return {
            "time": time.current_time(),
            "nodes": self.nodes,
            "radios": self.radios,
        }


def visualize(aggregators):
    num_aggregators = len(aggregators)
    fig, axes = plt.subplots(num_aggregators, 1, figsize=(10, 2 * num_aggregators))

    # If there's only one aggregator, wrap axes in a list for consistent handling
    if num_aggregators == 1:
        axes = [axes]

    for ax, radio_aggregator in zip(axes, aggregators):
        events = radio_aggregator.events
        states = radio_aggregator.states

        # Assign a unique number to each event name
        unique_names = sorted(
            set(event.name for event in events if isinstance(event, RadioEvent))
        )
        name_to_number = {
            name: i + 1 for i, name in enumerate(unique_names)
        }  # +1 to offset from radio states

        # Prepare data for plotting
        emitted_times = [
            event.time
            for event in events
            if isinstance(event, RadioEvent)
            and event.dir == RadioEventDirection.Outgoing
        ]
        received_times = [
            event.time
            for event in events
            if isinstance(event, RadioEvent)
            and event.dir == RadioEventDirection.Incoming
        ]

        emitted_y_positions = [
            name_to_number[event.name]
            for event in events
            if isinstance(event, RadioEvent)
            and event.dir == RadioEventDirection.Outgoing
        ]
        received_y_positions = [
            name_to_number[event.name]
            for event in events
            if isinstance(event, RadioEvent)
            and event.dir == RadioEventDirection.Incoming
        ]

        # Plot emitted and received signals
        ax.scatter(
            emitted_times,
            emitted_y_positions,
            color="blue",
            label="Emitted Signals",
            marker="^",
        )
        ax.scatter(
            received_times,
            received_y_positions,
            color="green",
            label="Received Signals",
            marker="v",
        )

        # Plot RadioStates
        state_times = [time for time, _ in states]
        state_values = [
            state - 1.5 for _, state in states
        ]  # Adjust y-level for visibility
        ax.step(
            state_times, state_values, where="post", label="Radio States", color="green"
        )

        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_yticks(range(len(unique_names) + 1))
        ax.set_yticklabels(["Radio States"] + unique_names)
        ax.set_title("Device")
        ax.legend()

    plt.tight_layout()
    plt.show()


def _reset():
    time.reset()
    np.random.seed(RAND_SEED)
    random.seed(RAND_SEED)


def run(
    signals: str,
    steps: int,
    network_lat_min: int,
    network_lat_max: int,
    radio_cooldown: int,
    local_batching: bool,
    recv_batching: bool,
    min_or_max: str,
):
    _reset()
    network = Network((network_lat_min, network_lat_max))  # ping latency

    radio_aggregator_a = RadioAggregator()
    radio_aggregator_b = RadioAggregator()
    radio_aggregator_c = RadioAggregator()
    radio_aggregator_d = RadioAggregator()
    radio_aggregator_e = RadioAggregator()
    radio_aggregator_f = RadioAggregator()

    node_a = Device("A", local_batching, recv_batching, min_or_max)
    node_b = Device("B", local_batching, recv_batching, min_or_max)
    node_c = Device("C", local_batching, recv_batching, min_or_max)
    node_d = Device("D", local_batching, recv_batching, min_or_max)
    node_e = Device("E", local_batching, recv_batching, min_or_max)
    node_f = Device("F", local_batching, recv_batching, min_or_max)

    import json

    signals: typing.Dict[str, typing.List[typing.Any]] = json.loads(signals)

    # TODO: yikes, this is ugly, but it works. This also has 3 nodes hardcoded, should be plenty for playing around
    for node_name in signals:
        src_node = None

        if node_name == "node_a":
            src_node = node_a
        elif node_name == "node_b":
            src_node = node_b
        elif node_name == "node_c":
            src_node = node_c
        elif node_name == "node_d":
            src_node = node_d
        elif node_name == "node_e":
            src_node = node_e
        elif node_name == "node_f":
            src_node = node_f
        else:
            raise ValueError(f"Unknown src node: {node_name}")

        signals_to_add: typing.List[Signal] = []
        for data in signals[node_name]:
            name = data["name"]
            period = int(data["period"])
            delta = period  # TODO

            dest_node = data["dest_node"]
            batch: bool = data.get("batch", False)

            dst_node = None
            if dest_node == "node_a":
                dst_node = node_a
            elif dest_node == "node_b":
                dst_node = node_b
            elif dest_node == "node_c":
                dst_node = node_c
            elif dest_node == "node_d":
                dst_node = node_d
            elif dest_node == "node_e":
                dst_node = node_e
            elif dest_node == "node_f":
                dst_node = node_f
            else:
                raise ValueError(f"Unknown dst node: {node_name}")

            signals_to_add.append(
                Signal(
                    name,
                    period=period,
                    delta=delta,
                    dest_node=dst_node,
                    batch=batch,
                    src_node=src_node,
                )
            )

        src_node.add_outgoing_signals(signals_to_add)

    radio_peer_a = Radio(node_a, radio_aggregator_a, network, radio_cooldown)
    radio_peer_b = Radio(node_b, radio_aggregator_b, network, radio_cooldown)
    radio_peer_c = Radio(node_c, radio_aggregator_c, network, radio_cooldown)
    radio_peer_d = Radio(node_d, radio_aggregator_d, network, radio_cooldown)
    radio_peer_e = Radio(node_e, radio_aggregator_e, network, radio_cooldown)
    radio_peer_f = Radio(node_f, radio_aggregator_f, network, radio_cooldown)

    node_a.add_radio(radio_peer_a)
    node_b.add_radio(radio_peer_b)
    node_c.add_radio(radio_peer_c)
    node_d.add_radio(radio_peer_d)
    node_e.add_radio(radio_peer_e)
    node_f.add_radio(radio_peer_f)

    stm = StateMachine()
    stm.add_network(network)
    stm.add_node(node_a)
    stm.add_node(node_b)
    stm.add_node(node_c)
    stm.add_node(node_d)
    stm.add_node(node_e)
    stm.add_node(node_f)

    stm.add_radio(radio_peer_a)
    stm.add_radio(radio_peer_b)
    stm.add_radio(radio_peer_c)
    stm.add_radio(radio_peer_d)
    stm.add_radio(radio_peer_e)
    stm.add_radio(radio_peer_f)

    for _ in range(steps):
        stm.update()

    # print total radio on time for each device
    def get_radio_high_time(radio_aggregator) -> int:
        total_radio_high_time = 0
        is_high = False

        for i in range(steps):
            if is_high:
                total_radio_high_time += 1

            for time, state in radio_aggregator.states:
                if time == i:
                    if state == RadioState.High:
                        is_high = True
                    else:
                        is_high = False

        return total_radio_high_time

    print(
        f"Total radio time for device A: {get_radio_high_time(radio_aggregator_a)} or {get_radio_high_time(radio_aggregator_a) / time.current_time():.2%} of total time"
    )
    print(
        f"Total radio time for device B: {get_radio_high_time(radio_aggregator_b)} or {get_radio_high_time(radio_aggregator_b) / time.current_time():.2%} of total time"
    )
    print(
        f"Total radio time for device C: {get_radio_high_time(radio_aggregator_c)} or {get_radio_high_time(radio_aggregator_c) / time.current_time():.2%} of total time"
    )
    print(
        f"Total radio time for device D: {get_radio_high_time(radio_aggregator_d)} or {get_radio_high_time(radio_aggregator_d) / time.current_time():.2%} of total time"
    )
    print(
        f"Total radio time for device E: {get_radio_high_time(radio_aggregator_e)} or {get_radio_high_time(radio_aggregator_e) / time.current_time():.2%} of total time"
    )
    print(
        f"Total radio time for device F: {get_radio_high_time(radio_aggregator_f)} or {get_radio_high_time(radio_aggregator_f) / time.current_time():.2%} of total time"
    )

    total = (
        get_radio_high_time(radio_aggregator_a)
        + get_radio_high_time(radio_aggregator_b)
        + get_radio_high_time(radio_aggregator_c)
        + get_radio_high_time(radio_aggregator_d)
        + get_radio_high_time(radio_aggregator_e)
        + get_radio_high_time(radio_aggregator_f)
    )

    percentage_total = total / (
        time.current_time() * 6
    )  # TODO: crude approximation, should be improved
    print(
        f"Total radio time accross all devices: {total} or {percentage_total:.2%} of total time"
    )

    visualize(
        [
            radio_aggregator_a,
            radio_aggregator_b,
            radio_aggregator_c,
            radio_aggregator_d,
            radio_aggregator_e,
            radio_aggregator_f,
        ]
    ),
