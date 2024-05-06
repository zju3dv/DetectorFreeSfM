from asyncio import Event
from typing import Tuple
import numpy as np
import random

import ray
from ray.actor import ActorHandle
from tqdm import tqdm


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter
    

class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

# Ray data utils
def chunks(lst, n, length=None):
    """Yield successive n-sized chunks from lst."""
    try:
        _len = len(lst)
    except TypeError as _:
        assert length is not None
        _len = length

    for i in range(0, _len, n):
        yield lst[i : i + n]
    # TODO: Check that lst is fully iterated

def chunks_balance(lst, n_split):
    if n_split == 0:
        # 0 is not allowed
        n_split = 1
    splited_list = [[] for i in range(n_split)]
    for id, obj in enumerate(lst):
        assign_id = id % n_split
        splited_list[assign_id].append(obj)
    return splited_list


def chunk_index(total_len, sub_len, shuffle=True):
    index_array = np.arange(total_len)
    if shuffle:
        random.shuffle(index_array)

    index_list = []
    for i in range(0, total_len, sub_len):
        index_list.append(list(index_array[i : i + sub_len]))
    
    return index_list

def chunk_index_balance(total_len, n_split, shuffle=True):
    index_array = np.arange(total_len)
    if shuffle:
        random.shuffle(index_array)

    splited_list = [[] for i in range(n_split)]
    for id, obj in enumerate(index_array):
        assign_id = id % n_split
        splited_list[assign_id].append(obj)
    return splited_list

def split_dict(_dict, n):
    for _items in chunks(list(_dict.items()), n):
        yield dict(_items)