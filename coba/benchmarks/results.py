import re
import collections

from numbers import Number
from operator import truediv
from itertools import chain, repeat, product, accumulate
from typing import Any, Iterable, Dict, List, Tuple, Optional, Sequence, Hashable, Iterator, Union, Type

from coba.config import CobaConfig
from coba.utilities import PackageChecker
from coba.pipes import Filter, Cartesian, JsonEncode, JsonDecode, StopPipe, Pipe, DiskSink, DiskSource 

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary: Sequence[str]):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._primary = primary
        self._columns = collections.OrderedDict(zip(primary,repeat(None)))

        self._rows_flat: Dict[Hashable, Dict[str,Any]] = {}
        self._rows_pack: Dict[Hashable, Dict[str,Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> Sequence[str]:
        return list(self._columns.keys())

    @property
    def dtypes(self) -> Sequence[Type[Union[int,float,bool,object]]]:

        flats = self._rows_flat
        packs = self._rows_pack
        keys  = self._rows_flat.keys()

        columns_packed = [ any([ col in packs[key] for key in keys]) for col in self.columns ]
        columns_values = [ [flats[key].get(col, packs[key].get(col, self._default(col))) for key in keys] for col in self.columns ]

        return [ self._infer_type(column_packed, column_values) for column_packed, column_values in zip(columns_packed,columns_values)]

    def to_pandas(self) -> Any:
        PackageChecker.pandas("Table.to_pandas")
        import pandas as pd #type: ignore
        import numpy as np #type: ignore #pandas installs numpy so if we have pandas we have numpy

        col_numpy = { col: np.empty(len(self), dtype=dtype) for col,dtype in zip(self.columns,self.dtypes)}

        index = 0

        for key in self._rows_flat.keys():

            flat = self._rows_flat[key]
            pack = self._rows_pack[key]

            size = 1 if not pack else len(pack['index'])

            for col in self.columns:
                if col in pack:
                    val = pack[col]

                elif col in flat:
                    if isinstance(flat[col], (tuple,list)):
                        val = [flat[col]]
                    else:
                        val = flat[col]

                else:
                    val = self._default(col)
                    
                col_numpy[col][index:(index+size)] = val

            index += size

        return pd.DataFrame(col_numpy, columns=self.columns)

    def to_tuples(self) -> Sequence[Tuple[Any,...]]:

        tooples = []

        for key in self._rows_flat.keys():
            
            flat = self._rows_flat[key]
            pack = self._rows_pack[key]

            if not pack:
                tooples.append(tuple(flat.get(col,self._default(col)) for col in self.columns))
            else:
                tooples.extend(zip(*[pack.get(col,repeat(flat.get(col,self._default(col)))) for col in self.columns]))
                
        return tooples

    def _default(self, column:str) -> Any:
        return [1] if column == "index" else float('nan')

    def _infer_type(self, is_packed: bool, values: Sequence[Any]) -> Type[Union[int,float,bool,object]]:

        types = []

        to_type = lambda value: None if value is None else type(value)

        for value in values:
            if is_packed and isinstance(value, (list,tuple)):
                types.extend([to_type(v) for v in value])
            else:
                types.append(to_type(value))
        
        return self._resolve_types(types)

    def _resolve_types(self, types: Sequence[Type[Any]]) -> Type[Union[int,float,bool,object]]:
        types = list(set(types))

        if len(types) == 1 and types[0] in [dict,str]:
            return object
        
        if len(types) == 1 and types[0] in [int,float,bool]:
            return types[0]

        if all(t in [None,int,float] for t in types):
            return float

        return object

    def _key(self, key: Union[Hashable, Sequence[Hashable]]) -> Union[Hashable,Tuple[Hashable,...]]:
        key_len = len(key) if isinstance(key,(list,tuple)) else 1
        
        assert key_len == len(self._primary), "Incorrect primary key length given"
        
        return tuple(key) if isinstance(key,(list,tuple)) else key

    def __iter__(self) -> Iterator[Dict[str,Any]]:
        for key in self._rows_flat.keys():
            yield self[key]

    def __contains__(self, key: Union[Hashable, Sequence[Hashable]]) -> bool:
        return self._key(key) in self._rows_flat

    def __str__(self) -> str:
        return str({"Table": self.name, "Columns": self.columns, "Rows": len(self)})

    def __len__(self) -> int:
        return sum([len(p['index']) if p else 1 for p in self._rows_pack.values()])

    def __setitem__(self, key: Union[Hashable, Sequence[Hashable]], values: Dict[str,Any]):

        key = self._key(key)

        #Try to make sure index is 3rd in order. 
        #It makes things look nicer in a data frame.
        if "_packed" in values: 
            self._columns["index"] = None

        row_flat = dict(zip(self._primary, key if isinstance(key,tuple) else [key]), **values)
        row_pack = row_flat.pop("_packed", {})

        if row_pack:
            assert len(set([len(value) for value in row_pack.values()])) == 1, "All packed columns must be equal length."

        if row_pack:
            row_pack['index'] = list(range(1,len(list(row_pack.values())[0])+1))

        self._rows_flat[key] = row_flat
        self._rows_pack[key] = row_pack
        self._columns.update(zip(list(row_flat.keys()) + list(row_pack.keys()), repeat(None)))

    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Dict[str,Any]:
        return dict(**self._rows_flat[self._key(key)], **self._rows_pack[self._key(key)])

    def __delitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> None:
        del self._rows_flat[self._key(key)]
        del self._rows_pack[self._key(key)]

class Result:
    """A class representing the result of a Benchmark evaluation on a given collection of Simulations and Learners."""

    @staticmethod
    def from_file(filename: str) -> 'Result':
        """Create a Result from a transaction file."""
        
        #Why is this here??? This is really confusing in practice
        #if filename is None or not Path(filename).exists(): return Result()

        json_encode = Cartesian(JsonEncode())
        json_decode = Cartesian(JsonDecode())

        Pipe.join(DiskSource(filename), [json_decode, ResultPromote(), json_encode], DiskSink(filename, 'w')).run()
        
        return Result.from_transactions(Pipe.join(DiskSource(filename), [json_decode]).read())

    @staticmethod
    def from_transactions(transactions: Iterable[Any]) -> 'Result':

        result = Result()

        for trx in transactions:
            if trx[0] == "version"  : result.version   = trx[1]
            if trx[0] == "benchmark": result.benchmark = trx[1]
            if trx[0] == "L"        : result._learners    [trx[1]       ] = trx[2]
            if trx[0] == "S"        : result._simulations [trx[1]       ] = trx[2]
            if trx[0] == "I"        : result._interactions[tuple(trx[1])] = trx[2]

        return result

    def __init__(self) -> None:
        """Instantiate a Result class."""

        self.version    : Optional[int]  = None
        self.benchmark  : Dict[str, Any] = {}

        #providing the types in advance makes to_pandas about 10 times faster since we can preallocate space
        self._interactions = Table("Interactions", ['simulation_id', 'learner_id'])
        self._learners     = Table("Learners"    , ['learner_id'])
        self._simulations  = Table("Simulations" , ['simulation_id'])

    @property
    def learners(self) -> Table:
        """The collection of learners evaluated by Benchmark. The easiest way to work with the 
            learners is to convert them to a pandas data frame via Result.learners.to_pandas()
        """
        return self._learners

    @property
    def simulations(self) -> Table:
        """The collection of simulations used to evaluate each learner in the Benchmark. The easiest
            way to work with simulations is to convert to a dataframe via Result.simulations.to_pandas()
        """
        return self._simulations

    @property
    def interactions(self) -> Table:
        """The collection of interactions that learners chose actions for in the Benchmark. Each interaction
            has a simulation_id and learner_id column to link them to the learners and simulations tables. The 
            easiest way to work with interactions is to convert to a dataframe via Result.interactions.to_pandas()
        """
        return self._interactions

    def plot_learners(self, 
        source_pattern :Union[str,int] = ".*",
        learner_pattern:Union[str,int] = ".*", 
        span:int = None,
        start:Union[int,float]=0.05,
        end:Union[int,float] = 1.,
        err_every:Union[int,float]=.05,
        err_type:str=None,
        figsize=(9,6),
        ax=None) -> None:
        """This plots the performance of multiple Learners on multiple simulations. It gives a sense of the expected 
            performance for different learners across independent simulations. This plot is valuable in gaining insight 
            into how various learners perform in comparison to one another. 

        Args:
            source_pattern: The pattern to match when determining which simulations to include in the plot. The "source" 
                matched against is either the "source" column in the simulations table or the first item in the list in 
                the simulation 'pipes' column. The simulations can be seen most easily by Result.simulations.to_pandas().
            learner_pattern: The pattern to match against the 'full_name' column in learners to determine which learners
                to include in the plot. In the case of multiple matches only the last match is kept. The learners table in
                Result can be examined via result.learners.to_pandas().
            span: In general this indicates how many previous evaluations to average together. In practice this works
                identically to ewm span value in the Pandas API. Additionally, if span equals None then all previous 
                rewards are averaged together and that value is plotted. Compare this to span = 1 WHERE only the current 
                reward is plotted for each interaction.
            start: Determines at which interaction the plot will start at. If start is greater than 1 we assume start is
                an interaction index. If start is less than 1 we assume start is the percent of interactions to skip
                before starting the plot.
            end: Determines at which interaction the plot will stop at. If end is greater than 1 we assume end is
                an interaction index. If end is less than 1 we assume end is the percent of interactions to end on.
            err_every: Determines frequency of bars indicating the standard deviation of the population should be drawn. 
                Standard deviation gives a sense of how well the plotted average represents the underlying distribution. 
                Standard deviation is most valuable when plotting against multiple simulations. If plotting against a single 
                simulation standard error may be a more useful indicator of confidence. The value for sd_every should be
                between 0 to 1 and will determine how frequently the standard deviation bars are drawn.
            err_type: Determines what the error bars are. Valid types are `None`, 'se', and 'sd'. If err_type is None then 
                plot will use SEM when there is only one source simulation otherwise it will use SD. Otherwise plot will
                display the standard error of the mean for 'se' and the standard deviation for 'sd'.
        """

        PackageChecker.matplotlib('Result.standard_plot')

        learner_ids    = []
        learner_names  = {}
        sources        = set()
        simulation_ids = []

        if isinstance(source_pattern, Number):
            source_pattern = f'(\D|^){source_pattern}(\D|$)'

        if isinstance(learner_pattern, Number):
            learner_pattern = f'(\D|^){learner_pattern}(\D|$)'

        for simulation in self._simulations:

            if 'source' in simulation:
                source = simulation['source']
            else:
                #this is a hack...
                source_end = max(simulation['pipe'].find("},{"), simulation['pipe'].find(","))
                source_end = source_end if source_end > -1 else len(simulation['pipe'])
                source     = simulation['pipe'][0:source_end]

            if re.search(source_pattern, source):
                sources.add(source)
                simulation_ids.append(simulation['simulation_id'])

        for learner in self._learners:
            if re.search(learner_pattern, learner['full_name']):
                learner_names[learner['learner_id']] = learner['full_name']
                learner_ids.append(learner['learner_id'])

        if len(learner_ids) == 0:
            CobaConfig.Logger.log(f"No learners were found matching {learner_pattern}")

        if len(simulation_ids) == 0:
            CobaConfig.Logger.log(f"No simulations were found with a source matching {source_pattern}")

        if len(learner_ids) == 0 or len(simulation_ids) == 0:
            return

        learner_ids = sorted(learner_ids, key=lambda id: learner_names[id])

        if err_type is None and len(sources) == 1: err_type = 'se'
        if err_type is None and len(sources) >= 2: err_type = 'sd'

        progressives: Dict[int,List[Sequence[float]]] = collections.defaultdict(list)

        for simulation_id, learner_id in product(simulation_ids,learner_ids):
            
            if (simulation_id,learner_id) not in self._interactions: continue

            rewards = self._interactions[(simulation_id,learner_id)]["reward"]
            if type(rewards[0]) is tuple:
                rewards = [reward[0] for reward in rewards]

            if span is None or span >= len(rewards):
                cumwindow  = list(accumulate(rewards))
                cumdivisor = list(range(1,len(cumwindow)+1))
            
            elif span == 1:
                cumwindow  = list(rewards)
                cumdivisor = [1]*len(cumwindow)

            else:
                alpha = 2/(1+span)
                cumwindow  = list(accumulate(rewards          , lambda a,c: c + (1-alpha)*a))
                cumdivisor = list(accumulate([1.]*len(rewards), lambda a,c: c + (1-alpha)*a)) #type: ignore

            progressives[learner_id].append(list(map(truediv, cumwindow, cumdivisor)))

        import matplotlib.pyplot as plt #type: ignore
        import numpy as np #type: ignore

        if not progressives:
            CobaConfig.Logger.log("No interaction data was found for plot_learners.")
            return
        
        full_figure = ax is None

        if full_figure:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1) #type: ignore

        for i,learner_id in enumerate(learner_ids):

            label = self._learners[learner_id]["full_name"]
            Z     = list(zip(*progressives[learner_id]))
            
            if not Z: continue

            N     = [ len(z) for z in Z        ]
            Y     = [ sum(z)/len(z) for z in Z ]
            X     = list(range(1,len(Y)+1))

            start_idx = int(start*len(X)) if start <  1 else int(start)
            end_idx   = int(end*len(X))   if end   <= 1 else int(end)

            end_idx   = len(X) if end_idx   > len(X) else end_idx
            start_idx = 0      if start_idx < 0      else start_idx

            if start_idx >= end_idx:
                CobaConfig.Logger.log("The plot's given end <= start making plotting impossible.")
                return

            X = X[start_idx:end_idx]
            Y = Y[start_idx:end_idx]
            Z = Z[start_idx:end_idx]

            if len(X) == 0: continue

            #this is much faster than python's native stdev
            #and more or less free computationally so we always
            #calculate it regardless of if they are showing them
            #we are using the identity Var[Y] = E[Y^2]-E[Y]^2
            Y2 = [ sum([zz**2 for zz in z])/len(z) for z in Z ]
            SD = [ (y2-y**2)**(1/2) for y,y2 in zip(Y,Y2)     ]
            SE = [ sd/(n**(1/2)) for sd,n in zip(SD,N)        ]

            err_every = int(len(X)*err_every) if err_every < 1 else err_every
            err_start = int(X[0] + i*len(X)*err_every**2) if err_every < 1 else err_every

            if not err_every:
               ax.plot(X, Y,label=label)
            else:
                yerr = SE if err_type.lower() == 'se' else SD #type: ignore
                ax.errorbar(X, Y, yerr=yerr, elinewidth=0.5, errorevery=(err_start,err_every), label=label)

        if full_figure:
            ax.set_xticks(np.clip(ax.get_xticks(), min(X), max(X)))
            ax.set_title (("Instantaneous" if span == 1 else "Progressive" if span is None else f"Span {span}") + " Reward")
            ax.set_ylabel("Reward")
            ax.set_xlabel("Interactions")

            #make room for the legend
            scale = 0.65
            box1 = ax.get_position()
            ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])

            # Put a legend below current axis
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .3), ncol=1, fontsize='medium') #type: ignore

            plt.show()

    def plot_shuffles(self, 
        source_pattern:str = ".*", 
        learner_pattern:str = ".*", 
        span:int=None,
        start:Union[int,float]=0.05,
        end:Union[int,float] = 1.,
        err_every:Union[int,float]=.05,
        err_type:str=None,
        figsize=(8,6)) -> None:
        """This plots the performance of a single Learner on multiple shuffles of the same source. It gives a sense of the
            variance in peformance for the learner on the given simulation source. This plot is valuable if looking for a 
            reliable learner on a fixed problem.

        Args:
            source_pattern: The pattern to match when determining which simulations to include in the plot. The "source" 
                matched against is either the "source" column in the simulations table or the first item in the list in 
                the simulation 'pipes' column. The simulations can be seen most easily by Result.simulations.to_pandas().
            learner_pattern: The pattern to match against the 'full_name' column in learners to determine which learners
                to include in the plot. In the case of multiple matches only the last match is kept. The learners table in
                Result can be examined via result.learners.to_pandas().
            span: In general this indicates how many previous evaluations to average together. In practice this works
                identically to ewm span value in the Pandas API. Additionally, if span equals None then all previous 
                rewards are averaged together and that value is plotted. Compare this to span = 1 WHERE only the current 
                reward is plotted for each interaction.
            start: Determines at which interaction the plot will start at. If start is greater than 1 we assume start is
                an interaction index. If start is less than 1 we assume start is the percent of interactions to skip
                before starting the plot.
            end: Determines at which interaction the plot will stop at. If end is greater than 1 we assume end is
                an interaction index. If end is less than 1 we assume end is the percent of interactions to end on.
            err_every: Determines frequency of bars indicating the standard deviation of the population should be drawn. 
                Standard deviation gives a sense of how well the plotted average represents the underlying distribution. 
                Standard deviation is most valuable when plotting against multiple simulations. If plotting against a single 
                simulation standard error may be a more useful indicator of confidence. The value for sd_every should be
                between 0 to 1 and will determine how frequently the standard deviation bars are drawn.
            err_type: Determines what the error bars are. Valid types are `None`, 'se', and 'sd'. If err_type is None then 
                plot will use SEM when there is only one source simulation otherwise it will use SD. Otherwise plot will
                display the standard error of the mean for 'se' and the standard deviation for 'sd'.

        """

        PackageChecker.matplotlib('Result.standard_plot')

        simulation_ids     = []
        simulation_sources = []
        learner_id         = None

        if isinstance(source_pattern, Number):
            source_pattern = f'(\D|^){source_pattern}(\D|$)'

        if isinstance(learner_pattern, Number):
            learner_pattern = f'(\D|^){learner_pattern}(\D|$)'
        
        for simulation in self._simulations:
            
            if 'source' in simulation:
                sim_source = simulation['source']
            else:
                #this is a hack...
                source_end = max(simulation['pipe'].find("},{"), simulation['pipe'].find(","))
                source_end = source_end if source_end > -1 else len(simulation['pipe'])
                sim_source = simulation['pipe'][0:source_end]

            if re.search(source_pattern, sim_source):
                simulation_ids.append(simulation['simulation_id'])
                simulation_sources.append(sim_source)

        for learner in self._learners:
            if re.search(learner_pattern,learner['full_name']):
                learner_id = learner['learner_id']

        progressives: List[Sequence[float]] = []

        if len(simulation_ids) == 0:
            CobaConfig.Logger.log(f"No simulation was found with a source matching '{source_pattern}' when executing `plot_shuffles`.")
            return

        if learner_id is None:
            CobaConfig.Logger.log(f"No learner was found who's fullname matched '{learner_pattern}' when executing `plot_shuffles`.")
            return

        for simulation_id in simulation_ids:
            
            if (simulation_id,learner_id) not in self._interactions: continue

            rewards = self._interactions[(simulation_id,learner_id)]["reward"]

            if span is None or span >= len(rewards):
                cumwindow  = list(accumulate(rewards))
                cumdivisor = list(range(1,len(cumwindow)+1))

            elif span == 1:
                cumwindow  = list(rewards)
                cumdivisor = [1]*len(cumwindow)

            else:
                cumwindow  = list(accumulate(rewards))
                cumwindow  = cumwindow + [0] * span
                cumwindow  = [ cumwindow[i] - cumwindow[i-span] for i in range(len(cumwindow)-span) ]
                cumdivisor = list(range(1, span)) + [span]*(len(cumwindow)-span+1)

            progressives.append(list(map(truediv, cumwindow, cumdivisor)))

        if not progressives:
            CobaConfig.Logger.log("No interaction data was found for the plot_shuffles.")
            return

        import matplotlib.pyplot as plt #type: ignore
        import numpy as np #type: ignore

        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(1,1,1) #type: ignore

        color = next(ax._get_lines.prop_cycler)['color']

        for shuffle in progressives:

            Y     = shuffle
            X     = list(range(1,len(Y)+1))

            start_idx = int(start*len(X)) if start <  1 else int(start)
            end_idx   = int(end*len(X))   if end   <= 1 else int(end)

            end_idx   = len(X) if end_idx   > len(X) else end_idx
            start_idx = 0      if start_idx < 0      else start_idx

            if start_idx >= end_idx:
                CobaConfig.Logger.log("The plot's given end <= start making plotting impossible.")
                return

            X = X[start_idx:end_idx]
            Y = Y[start_idx:end_idx]

            ax.plot(X, Y, label='_nolegend_', color=color, alpha=0.15)

        plt.gca().set_prop_cycle(None)
        self.plot_learners(source_pattern, learner_pattern, span=span, start=start, end=end, err_every=err_every, err_type=err_type, ax=ax)

        ax.set_xticks(np.clip(ax.get_xticks(), min(X), max(X)))

        simulation_sources = list(set(simulation_sources))
        source = simulation_sources[0] if len(simulation_sources) == 1 else str(simulation_sources)

        ax.set_title (("Instantaneous" if span == 1 else "Progressive" if span is None else f"Span {span}") + f" Reward for '{source}'")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Interactions")

        #make room for the legend
        scale = 0.85
        box1 = ax.get_position()
        ax.set_position([box1.x0, box1.y0 + box1.height * (1-scale), box1.width, box1.height * scale])

        #Put a legend below current axis
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .1), ncol=1, fontsize='medium') #type: ignore

        plt.show()

    def __str__(self) -> str:
        return str({ "Learners": len(self._learners), "Simulations": len(self._simulations), "Interactions": len(self._interactions) })

    def __repr__(self) -> str:
        return str(self)

class ResultPromote(Filter):

    CurrentVersion = 3

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        items_iter = iter(items)
        items_peek = next(items_iter)
        items_iter = chain([items_peek], items_iter)

        version = 0 if items_peek[0] != 'version' else items_peek[1]

        if version == ResultPromote.CurrentVersion:
            raise StopPipe()

        while version != ResultPromote.CurrentVersion:
            if version == 0:
                promoted_items = [["version",1]]

                for transaction in items:

                    if transaction[0] == "L":

                        index  = transaction[1][1]['learner_id']
                        values = transaction[1][1]

                        del values['learner_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "S":

                        index  = transaction[1][1]['simulation_id']
                        values = transaction[1][1]

                        del values['simulation_id']

                        promoted_items.append([transaction[0], index, values])

                    if transaction[0] == "B":
                        key_columns = ['learner_id', 'simulation_id', 'seed', 'batch_index']
                        
                        index  = [ transaction[1][1][k] for k in key_columns ]
                        values = transaction[1][1]
                        
                        for key_column in key_columns: del values[key_column]
                        
                        if 'reward' in values:
                            values['reward'] = values['reward'].estimate
                        
                        if 'mean_reward' in values:
                            values['reward'] = values['mean_reward'].estimate
                            del values['mean_reward']

                        values['reward'] = round(values['reward', 5])

                        promoted_items.append([transaction[0], index, values])

                items   = promoted_items
                version = 1

            if version == 1:

                n_seeds       : Optional[int]                  = None
                S_transactions: Dict[int, Any]                 = {}
                S_seeds       : Dict[int, List[Optional[int]]] = collections.defaultdict(list)

                B_rows: Dict[Tuple[int,int], Dict[str, List[float]] ] = {}
                B_cnts: Dict[int, int                               ] = {}

                promoted_items = [["version",2]]

                for transaction in items:

                    if transaction[0] == "benchmark":
                        n_seeds = transaction[1].get('n_seeds', None)

                        del transaction[1]['n_seeds']
                        del transaction[1]['batcher']
                        del transaction[1]['ignore_first']

                        promoted_items.append(transaction)

                    if transaction[0] == "L":
                        promoted_items.append(transaction)

                    if transaction[0] == "S":
                        S_transactions[transaction[1]] == transaction

                    if transaction[0] == "B":
                        S_id = transaction[1][1]
                        seed = transaction[1][2]
                        L_id = transaction[1][0]
                        B_id = transaction[1][3]
                        
                        if n_seeds is None:
                            raise StopPipe("We are unable to promote logs from version 1 to version 2")

                        if seed not in S_seeds[S_id]:
                            S_seeds[S_id].append(seed)
                            
                            new_S_id = n_seeds * S_id + S_seeds[S_id].index(seed)
                            new_dict = S_transactions[S_id][2].clone()
                            
                            new_dict["source"]  = str(S_id)
                            new_dict["filters"] = f'[{{"Shuffle":{seed}}}]'

                            B_cnts[S_id] = new_dict['batch_count']

                            promoted_items.append(["S", new_S_id, new_dict])

                        if B_id == 0: B_rows[(S_id, L_id)] = {"N":[], "reward":[]}

                        B_rows[(S_id, L_id)]["N"     ].append(transaction[2]["N"])
                        B_rows[(S_id, L_id)]["reward"].append(transaction[2]["reward"])

                        if len(B_rows[(S_id, L_id)]["N"]) == B_cnts[S_id]:
                            promoted_items.append(["B", [S_id, L_id], B_rows[(S_id, L_id)]])
                            del B_rows[(S_id, L_id)]

                items   = promoted_items
                version = 2

            if version == 2:

                promoted_items = [["version",3]]

                for transaction in items:
                    
                    #upgrade all reward entries to the packed format which will now allow array types and dict types.
                    if transaction[0] == "B":
                        rewards = transaction[2]["reward"]
                        del transaction[2]["reward"]
                        transaction[2]["_packed"] = {"reward": rewards}
                    
                    #Change from B to I to be consistent with result property name: `interactions`
                    if transaction[0] == "B": 
                        transaction[0] = "I"
                    
                    promoted_items.append(transaction)

                items   = promoted_items
                version = 3

        return items
