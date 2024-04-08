import random
import typing
from typing import List, TypedDict

import matplotlib.pyplot as plt
import numpy as np

RAND_SEED = 42

# TODO move outside to interactive sliders
MIN_SIGNAL_TS = 40
MAX_SIGNAL_TS = 160
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
    def __init__(self, name: str):
        self.events: typing.List[RadioEvent] = []
        self.states: typing.List[typing.Tuple[int, int]] = []
        self._name = name

    def get_name(self) -> str:
        return self._name

    def add_radio_state(self, radio_state: int, time: int) -> None:
        self.states.append((time, radio_state))

    def add_event(self, event: RadioEvent) -> None:
        self.events.append(event)

    def __repr__(self):
        return f"RadioAggregator: {self.events} {self.states} {self._name}"


# TODO: there are limitations in that all numbers are in seconds(assumed amounts). We should multiply everything by 1000 to simulate milliseconds, or at least 10-100
class Time:
    def __init__(self):
        self.reset()

    def current_time(self) -> int:
        return self._curr_time

    def advance_time(self) -> None:
        self._curr_time += 1

    def reset(self):
        # TODO nonzero to simplify the example as we don't need to check against zero
        self._curr_time = 1

    def __repr__(self):
        return f"Time: {self._curr_time}"


time = Time()


class IncomingSignal:
    def __init__(self, name: str, received_time_ts: int, src_node: "Device"):
        self._name = name
        self._received_time_ts = received_time_ts
        self._src_node = src_node

    def __repr__(self):
        return (
            f"IncomingSignal: {self._name}, {self._received_time_ts}, {self._src_node}"
        )


class Signal:
    def __init__(
        self,
        name: str,
        period: int,
        dest_node=None,
        src_node=None,
        batch: bool = False,
    ):
        self._first_emit = True
        self._src_node = src_node
        self._name = name
        self._period = period
        self._dest_node = dest_node
        self._batch = batch

        # signal might not be initiated immediately, so immitate some delay to misalign
        self._emit_time = time.current_time() + random.randint(
            MIN_SIGNAL_TS, MAX_SIGNAL_TS
        )

    def get_dest_node_name(self) -> str:
        return self._dest_node.get_name()

    def post_emit(self, time: int, gcd: int, premature: bool) -> None:
        assert time <= self._emit_time

        if self._first_emit and premature:
            self._emit_time = time + gcd
            self._first_emit = False
        else:
            self._emit_time = time + self._period

    def can_emit_at(self, time: int) -> bool:
        return self._emit_time == time

    def convert_to_incoming(self) -> IncomingSignal:
        assert self._src_node is not None
        return IncomingSignal(self._name, time.current_time(), self._src_node)

    def __repr__(self):
        return (
            f"Signal: {self._name}, {self._period}, {self._emit_time} {self._dest_node}"
        )


class Device:
    def __init__(self, name: str, local_batching: bool, recv_batching: bool):
        self._name = name
        self._radio: typing.Optional[Radio] = None
        self._outgoing_signals: typing.Optional[list[Signal]] = None

        self._local_batching = local_batching
        self._recv_batching = recv_batching

        self._peer_last_received: typing.Dict[str, bool] = {}

    def get_name(self) -> str:
        return self._name

    def add_radio(self, radio) -> None:
        assert self._radio is None

        self._radio = radio

    def add_outgoing_signals(self, signals: list[Signal]) -> None:
        self._outgoing_signals = signals

    def receive(self, signal: IncomingSignal) -> None:
        assert self._radio is not None

        self._peer_last_received[signal._src_node._name] = True

    def emit(self, signal: Signal, gcd: int, premature: bool) -> None:
        signal.post_emit(time.current_time(), gcd, premature)

        assert self._radio is not None
        self._radio.emit(signal)

        self._peer_last_received[signal.get_dest_node_name()] = False

    def update(self) -> None:
        if self._outgoing_signals is None:
            return

        assert self._radio
        curr_time = time.current_time()
        gcd = min([signal._period for signal in self._outgoing_signals])

        if self._recv_batching:
            # when receive batching is enabled, it means this is a good opportunity
            # to emit data towards the peer from which data came. There's a high chance
            # that peers radio is hot however we're very sure our is hot so it makes sense to use that.
            # this assumes that the cost of sending data when radio is hot is insignificant
            for signal in self._outgoing_signals:
                if (
                    signal._dest_node.name in self._peer_last_received
                    and self._peer_last_received[signal._dest_node.name]
                ):
                    pass
                    # TODO

        elif self._local_batching:
            # the idea behind gcd is to find lowest common interval and align on that.
            # this assumes that all signals are sharing a common period or else
            # this doesn't work.
            # Signal must hold state if it was already emitted or not. Because we are
            # guaranteed that all periods are multiplier of gcd, we can just wait until
            # alignment time and simply emit the signal. This will guarantee that this signal
            # is immediately aligned with all other signals.
            # Afterwards once the signal is marked as emitted first time, we can proceed as usual
            # and perform regular emission based on the period of the signal so it will naturally
            # be emitted on it's gcd*N time.
            gcd = min([signal._period for signal in self._outgoing_signals])
            for signal in self._outgoing_signals:
                if curr_time % gcd == 0 and signal._first_emit:
                    self.emit(signal, gcd, True)

        # make sure we emit on hard deadline of a signal
        # this is crucial to call at the very end to guarantee emission of events
        for signal in self._outgoing_signals:
            if signal.can_emit_at(curr_time):
                self.emit(signal, gcd, False)

    def __repr__(self):
        return f"Device: {self._name}"


class Network:  #
    def __init__(self, latency: tuple[int, int]):
        assert latency[0] < latency[1]

        self._latency = latency

        # ingress queue is used to simulate the sending of packets. packet is stored with timestamp and then sent to the deveice once the time arrives
        self.ingress_queue: typing.List[
            typing.Tuple[int, typing.Tuple[Device, IncomingSignal]]
        ] = []

    def set_latency(self, latency: tuple[int, int]) -> None:
        assert latency[0] > latency[1]

        self._latency = latency

    def send(self, dest_node: Device, signal: Signal) -> None:
        lat = np.random.randint(self._latency[0], self._latency[1] + 1)

        self.ingress_queue.append(
            (int(time.current_time() + lat), (dest_node, signal.convert_to_incoming()))
        )

    def update(self) -> None:
        for t, (dest_node, signal) in self.ingress_queue.copy():
            if t == time.current_time():
                assert dest_node._radio
                # TODO: can this be simulated nicer?
                dest_node._radio.receive(signal)
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
        self._state = RadioState.Low
        self._network = network
        self._cooldown = cooldown
        self._state_change_countdown = cooldown
        self._node = node
        self._aggregator = aggregator
        self._aggregator.add_radio_state(self._state, time.current_time())

    def _set_to_high(self) -> None:
        self._state = RadioState.High
        self._aggregator.add_radio_state(self._state, time.current_time())
        self._state_change_countdown = self._cooldown

    def receive(self, signal: IncomingSignal) -> None:
        self._set_to_high()
        self._aggregator.add_event(
            RadioEvent(time.current_time(), signal._name, RadioEventDirection.Incoming)
        )
        self._node.receive(signal)

    def emit(self, signal: Signal) -> None:
        self._set_to_high()
        self._aggregator.add_event(
            RadioEvent(time.current_time(), signal._name, RadioEventDirection.Outgoing)
        )

        self._network.send(signal._dest_node, signal)

    def update(self) -> None:
        if self._state == RadioState.High:
            if self._state_change_countdown > 0:
                self._state_change_countdown -= 1
            else:
                self._state = RadioState.Low
                self._aggregator.add_radio_state(self._state, time.current_time())

    def __repr__(self):
        return f"Radio: {self._state}, {self._state_change_countdown}"


class StateInfo(TypedDict):
    time: int
    nodes: List[Device]
    radios: List[Radio]


class StateMachine:
    def __init__(self):
        self._network = None
        self._nodes = []
        self._radios = []

    def add_network(self, network: Network) -> None:
        self._network = network

    def add_node(self, node: Device) -> None:
        self._nodes.append(node)

    def add_radio(self, radio: Radio) -> None:
        self._radios.append(radio)

    def update(self) -> None:
        self._network.update()

        for node in self._nodes:
            node.update()

        for radio in self._radios:
            radio.update()

        time.advance_time()

    # return all of the state
    def state(self) -> StateInfo:
        return {
            "time": time.current_time(),
            "nodes": self._nodes,
            "radios": self._radios,
        }


@typing.no_type_check
def visualize(aggregators: typing.List[RadioAggregator]):

    # filter empty aggregators
    aggregators = [a for a in aggregators if len(a.events) > 0]

    num_aggregators = len(aggregators)
    fig, axes = plt.subplots(num_aggregators, 1, figsize=(10, 3 * num_aggregators))

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
            state - 0.5 for _, state in states
        ]  # Adjust y-level for visibility
        ax.step(
            state_times, state_values, where="post", label="Radio States", color="green"
        )

        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_yticks(range(len(unique_names) + 1))
        ax.set_yticklabels(["Radio States"] + unique_names)
        ax.set_title(f"Device {radio_aggregator.get_name()}")

        ax.legend()

    plt.tight_layout()
    plt.show()


def _reset():
    time.reset()
    np.random.seed(RAND_SEED)
    random.seed(RAND_SEED)


def run(
    signals_cfg: str,
    steps: int,
    network_lat_min: int,
    network_lat_max: int,
    radio_cooldown: int,
    local_batching: bool,
    recv_batching: bool,
):
    _reset()
    network = Network((network_lat_min, network_lat_max))  # ping latency

    radio_aggregator_a = RadioAggregator("A")
    radio_aggregator_b = RadioAggregator("B")
    radio_aggregator_c = RadioAggregator("C")
    radio_aggregator_d = RadioAggregator("D")
    radio_aggregator_e = RadioAggregator("E")
    radio_aggregator_f = RadioAggregator("F")

    node_a = Device("A", local_batching, recv_batching)
    node_b = Device("B", local_batching, recv_batching)
    node_c = Device("C", local_batching, recv_batching)
    node_d = Device("D", local_batching, recv_batching)
    node_e = Device("E", local_batching, recv_batching)
    node_f = Device("F", local_batching, recv_batching)

    import json

    signals: typing.Dict[str, typing.List[typing.Any]]
    signals = json.loads(signals_cfg)

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

    if radio_aggregator_a.events:
        print(
            f"Total radio time for device A: {get_radio_high_time(radio_aggregator_a)} or {get_radio_high_time(radio_aggregator_a) / time.current_time():.2%} of total time"
        )
    if radio_aggregator_b.events:
        print(
            f"Total radio time for device B: {get_radio_high_time(radio_aggregator_b)} or {get_radio_high_time(radio_aggregator_b) / time.current_time():.2%} of total time"
        )
    if radio_aggregator_c.events:
        print(
            f"Total radio time for device C: {get_radio_high_time(radio_aggregator_c)} or {get_radio_high_time(radio_aggregator_c) / time.current_time():.2%} of total time"
        )
    if radio_aggregator_d.events:
        print(
            f"Total radio time for device D: {get_radio_high_time(radio_aggregator_d)} or {get_radio_high_time(radio_aggregator_d) / time.current_time():.2%} of total time"
        )
    if radio_aggregator_e.events:
        print(
            f"Total radio time for device E: {get_radio_high_time(radio_aggregator_e)} or {get_radio_high_time(radio_aggregator_e) / time.current_time():.2%} of total time"
        )
    if radio_aggregator_f.events:
        print(
            f"Total radio time for device F: {get_radio_high_time(radio_aggregator_f)} or {get_radio_high_time(radio_aggregator_f) / time.current_time():.2%} of total time"
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
