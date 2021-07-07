import collections
from typing import Any, Iterable, Optional, Sequence

from coba.learners import Learner
from coba.pipes import Pipe, Filter, Source, Sink, Cartesian, JsonEncode, DiskSink, MemorySink
from coba.simulations import Simulation, Take, Shuffle

from coba.benchmarks.results import Result, ResultPromote

class Transaction:

    @staticmethod
    def version(version = None) -> Any:
        return ['version', version or ResultPromote.CurrentVersion]

    @staticmethod
    def benchmark(n_learners, n_simulations) -> Any:
        data = {
            "n_learners"   : n_learners,
            "n_simulations": n_simulations,
        }

        return ['benchmark',data]

    @staticmethod
    def learner(learner_id:int, **kwargs) -> Any:
        """Write learner metadata row to Result.
        
        Args:
            learner_id: The primary key for the given learner.
            kwargs: The metadata to store about the learner.
        """
        return ["L", learner_id, kwargs]

    @staticmethod
    def learners(learners: Sequence[Learner]) -> Iterable[Any]:
        for index, learner in enumerate(learners):

            try:
                params = learner.params
            except:
                params = {}

            try:
                family = learner.family
            except:
                family = type(learner).__name__

            if len(learner.params) > 0:
                full_name = f"{learner.family}({','.join(f'{k}={v}' for k,v in learner.params.items())})"
            else:
                full_name = learner.family

            yield Transaction.learner(index, full_name=full_name, family=family, **params)

    @staticmethod
    def simulation(simulation_id: int, **kwargs) -> Any:
        """Write simulation metadata row to Result.
        
        Args:
            simulation_index: The index of the simulation in the benchmark's simulations.
            kwargs: The metadata to store about the learner.
        """
        return ["S", simulation_id, kwargs]

    @staticmethod
    def simulations(simulations:Sequence[Source[Simulation]]) -> Iterable[Any]:

        def get_source(simulation: Source[Simulation]) -> Source[Simulation]:

            if isinstance(simulation, Pipe.SourceFilters):
                return get_source(simulation._source)
            else:
                return simulation

        def get_filters(pipes: Source[Simulation]) -> Sequence[Filter[Simulation,Simulation]]:
            if isinstance(simulation, Pipe.SourceFilters):
                return simulation._filter._filters #we know these are flattened already
            else:
                return []

        for index, simulation in enumerate(simulations):

            source  = str(get_source(simulation)).strip('"')
            shuffle = "None"
            take    = "None"

            for filter in get_filters(simulation):
                if isinstance(filter, Shuffle): shuffle = str(filter._seed )
                if isinstance(filter, Take   ): take    = str(filter._count)

            yield Transaction.simulation(index, source=source, shuffle=shuffle, take=take, pipe=str(simulation))

    @staticmethod
    def interactions(simulation_id:int, learner_id:int, **kwargs) -> Any:
        """Write interaction evaluation metadata row to Result.

        Args:
            learner_id: The primary key for the learner we observed on the interaction.
            simulation_id: The primary key for the simulation the interaction came from.
            kwargs: The metadata to store about the interaction with the learner.
        """

        return ["I", (simulation_id, learner_id), kwargs]

class TransactionIsNew(Filter):

    def __init__(self, existing: Result):

        self._existing = existing

    def filter(self, transactions: Iterable[Any]) -> Iterable[Any]:
        
        for transaction in transactions:
            
            tipe  = transaction[0]

            if tipe == "version" and self._existing.version is not None:
                continue
            
            if tipe == "benchmark" and len(self._existing.benchmark) != 0:
                continue

            if tipe == "I" and transaction[1] in self._existing._interactions:
                continue

            if tipe == "S" and transaction[1] in self._existing._simulations:
                continue

            if tipe == "L" and transaction[1] in self._existing._learners:
                continue

            yield transaction

class TransactionSink(Sink):

    def __init__(self, transaction_log: Optional[str], restored: Result) -> None:

        json_encode = Cartesian(JsonEncode())

        final_sink = Pipe.join([json_encode], DiskSink(transaction_log)) if transaction_log else MemorySink()
        self._sink = Pipe.join([TransactionIsNew(restored)], final_sink)

    def write(self, items: Sequence[Any]) -> None:
        self._sink.write(items)

    @property
    def result(self) -> Result:
        if isinstance(self._sink, Pipe.FiltersSink):
            final_sink = self._sink.final_sink()
        else:
            final_sink = self._sink

        if isinstance(final_sink, MemorySink):
            return Result.from_transactions(final_sink.items)

        if isinstance(final_sink, DiskSink):
            return Result.from_file(final_sink.filename)

        raise Exception("Transactions were written to an unrecognized sink.")
