# container_env_balance_only.py

import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class Container:
    id: str
    weight: int
    destination: int  # smaller number = earlier destination

@dataclass
class State:
    yard: Tuple[Tuple[Container, ...], ...]   # stacks in yard
    left: Tuple[Container, ...]               # ship left side stack
    right: Tuple[Container, ...]              # ship right side stack
    g: int                                    # cost so far
    h: int                                    # heuristic estimate
    parent: Optional['State'] = None          # backpointer
    action: Optional[str] = None              # action description

class ContainerEnv:
    def __init__(self, yard_stacks: List[List[Container]], imbalance_bound: int):
        self.initial = State(
            yard=tuple(tuple(stack) for stack in yard_stacks),
            left=(),
            right=(),
            g=0,
            h=0
        )
        self.heuristic_type = "admissible"
        self.imbalance_bound = imbalance_bound

    def is_goal(self, state: State) -> bool:
        return all(len(stack) == 0 for stack in state.yard)

    def imbalance(self, state: State) -> int:
        left_w = sum(c.weight for c in state.left)
        right_w = sum(c.weight for c in state.right)
        return abs(left_w - right_w)

    def unloading_cost_estimate(self, state: State) -> int:
        def stack_cost(stack):
            cost = 0
            for i in range(len(stack)):
                for j in range(i+1, len(stack)):
                    if stack[i].destination > stack[j].destination:
                        cost += 2  # penalty for blocking
            return cost

        return stack_cost(state.left) + stack_cost(state.right)

    def unloading_cost_estimate_admissible(self, state: State) -> int:
            # This heuristic is admissible for the following reason:
            # For each container in the stack that blocks a container below it with an earlier destination,
            # we add a penalty of 2 and immediately break, so only the first blocking event per container is counted.
            # Thus, for each blocking configuration, we count at least one move needed to remove that blocking.
            # However, in the real cost calculation (see actual_unloading_cost), unblocking may require more moves
            # or fewer moves if a single move can resolve multiple blocks, but never fewer than what is counted here.
            # Therefore, this heuristic never overestimates the minimal number of moves required to unblock all containers,
            # making it admissible (it is a lower bound on the actual cost).
            def stack_cost(stack):
                cost = 0
                for i in range(len(stack)):
                    for j in range(i+1, len(stack)):
                        if stack[i].destination > stack[j].destination:
                            cost += 2  # penalty for blocking
                            break  # Once blocked, don't check lower containers
                return cost
            return stack_cost(state.left) + stack_cost(state.right)


    def heuristic(self, state: State) -> int:
        if self.heuristic_type == "admissible":
            return self.unloading_cost_estimate_admissible(state)
        elif self.heuristic_type == "non_admissible":
            return self.unloading_cost_estimate(state)
        else:
            raise ValueError(f"Invalid heuristic type: {self.heuristic_type}")


    def relocation_cost(self,containers: List[Container]):
        # """
        # Compute minimum relocation cost for a single stack.

        # stack: destinations from TOP -> BOTTOM.

        # At each destination:
        #   1. Remove containers blocking containers of that destination.
        #   2. Remove all containers of that destination.
        #   3. Reorder the temporarily removed containers optimally.
        #   4. Put them back.

        # Returns:
        #     total number of container relocations.
        # """

        stack = list([c.destination for c in containers])
        cost = 0

        # print(f"stack: {stack}")

        for destination in sorted(set(stack)):

            # Find the last occurrence of this destination.
            # Everything above it must be removed in order to
            # unload all containers of this destination.
            last = max(
                i for i, x in enumerate(stack)
                if x == destination
            )

            # Containers above the last destination container
            # that are NOT themselves destination containers
            # have to be relocated.
            relocated = [
                x for x in stack[:last]
                if x != destination
            ]

            cost += len(relocated)

            # Containers below the last destination container
            # remain untouched.
            below = stack[last + 1:]

            # The destination containers themselves are unloaded,
            # as are the relocated containers temporarily.
            #
            # Reorder relocated containers so that earlier
            # destinations are above later destinations.
            relocated.sort()

            stack = relocated + below

        # print(f"cost: {cost*2}")

        return cost*2
    
    def actual_unloading_cost(self,state: State) -> int:
        """
        # Compute the true unloading cost for the given state as the minimum number of "relocations" 
        # needed to unload all containers in order of increasing destination. 
        # For each stack, this is the minimum number of moves required so that no container with a later 
        # destination is above a container with an earlier destination when unloading,
        # i.e., number of times a container blocks another with an earlier destination below it.
        # One move can unblock multiple containers in a stack.
        # We compute this by counting, for each stack, the minimal number of reorders to sort destinations, 
        # i.e., len(stack) - length of Longest Increasing Subsequence (LIS) among destinations.
        of containers at each destination (i.e., minimizing blocking).
        Equivalent to: for each stack, number of containers - LIS by destination.
        """
        def lis_length(seq):
            # bisect_left is a function from the bisect module that finds the insertion point for x in a sorted list to maintain sorted order.
            # It is commonly used in the LIS (Longest Increasing Subsequence) algorithm to efficiently determine where to place elements.
            from bisect import bisect_left
            lis = []
            for x in seq:
                # bisect_left(lis, x) returns the leftmost index to insert x so that lis remains sorted
                pos = bisect_left(lis, x)
                if pos == len(lis):
                    lis.append(x)
                else:
                    lis[pos] = x
            print(f"{seq}:{seq}, lis:{lis}, cost:{len()- len(lis)}")
            return len(lis)

        cost = 0
        for stack in [state.left, state.right]:
            dest_seq = [c.destination for c in stack]
            if dest_seq:
                cost += len(dest_seq) - lis_length(dest_seq)
        return cost*2 # cost of an ublock is 2 since the blocking container needs to be moved to the top of the stack

    def neighbors(self, state: State) -> List[State]:
        successors = []
        for s_idx, stack in enumerate(state.yard):
            if not stack:
                continue
            c = stack[-1]  # top container

            # Try left side
            new_left = state.left + (c,)
            new_yard = list(list(st) for st in state.yard)
            new_yard[s_idx].pop()
            new_state = State(
                yard=tuple(tuple(st) for st in new_yard),
                left=new_left,
                right=state.right,
                g=state.g + 1,
                h=0,
                parent=state,
                action=f"Move {c.id} to LEFT"
            )
            if self.imbalance(new_state) <= self.imbalance_bound:
                new_state.h = self.heuristic(new_state)
                successors.append(new_state)

            # Try right side
            new_right = state.right + (c,)
            new_yard = list(list(st) for st in state.yard)
            new_yard[s_idx].pop()
            new_state = State(
                yard=tuple(tuple(st) for st in new_yard),
                left=state.left,
                right=new_right,
                g=state.g + 1,
                h=0,
                parent=state,
                action=f"Move {c.id} to RIGHT"
            )
            if self.imbalance(new_state) <= self.imbalance_bound:
                new_state.h = self.heuristic(new_state)
                successors.append(new_state)

        return successors


# ------------------------------
# Generic Beam Search
# ------------------------------
def beam_search(env: ContainerEnv, beam_width: int = 5) -> Optional[State]:
    frontier = [env.initial]
    visited = set()

    while frontier:
        new_frontier = []
        for state in frontier:
            if env.is_goal(state):
                return state, len(visited)
            for succ in env.neighbors(state):
                key = (succ.yard, succ.left, succ.right)
                if key not in visited:
                    visited.add(key)
                    new_frontier.append(succ)
        new_frontier.sort(key=lambda s: s.g + s.h)
        frontier = new_frontier[:beam_width]
    return None


# ------------------------------
# Generic A* Search
# ------------------------------
def astar_search(env: ContainerEnv) -> Optional[State]:
    counter = 0
    frontier = [(env.initial.g + env.initial.h, counter, env.initial)]
    visited = set()
    while frontier:
        f, _, state = heapq.heappop(frontier)
        if env.is_goal(state):
            return state, len(visited)
        key = (state.yard, state.left, state.right)
        if key in visited:
            continue
        visited.add(key)
        for succ in env.neighbors(state):
            counter += 1
            heapq.heappush(frontier, (succ.g + succ.h, counter, succ))
    return None

   

# ------------------------------
# Helper to reconstruct plan
# ------------------------------
def reconstruct_plan(goal: State) -> List[str]:
    plan = []
    s = goal
    while s.parent is not None:
        plan.append(s.action)
        s = s.parent
    return list(reversed(plan))


# ------------------------------
# Example usage
# ------------------------------
def print_state(label: str, state: State, env: ContainerEnv):
    print(f"\n--- {label} ---")
    print("Yard:")
    for i, st in enumerate(state.yard):
        print(f"  Stack {i+1}: {[(c.id,c.weight,c.destination) for c in st]}")
    print(f"Left ship: {[(c.id,c.weight,c.destination) for c in state.left]}")
    print(f"Right ship: {[(c.id,c.weight,c.destination) for c in state.right]}")
    print(f"Loading cost (g): {state.g}")
    unload_cost = env.unloading_cost_estimate(state)
    # actual_unload = env.actual_unloading_cost(state)
    print(f"Unloading cost (h est): {unload_cost}")
    # print(f"Actual extra unloading cost:{actual_unload}")
    print(f"Total cost: {state.g + unload_cost}")


import random

def random_instance(num_containers: int, num_stacks: int, imbalance_bound: int = 5,
                    max_weight: int = 10, max_destination: int = 5) -> ContainerEnv:
    """
    Generate a random container loading instance.
    - num_containers: total number of containers
    - num_stacks: number of yard stacks
    - imbalance_bound: allowed weight difference between ship sides
    - max_weight: maximum container weight
    - max_destination: maximum destination id
    """
    containers = [
        Container(f"C{i+1}", random.randint(1, max_weight), random.randint(1, max_destination))
        for i in range(num_containers)
    ]

    # Randomly distribute into yard stacks
    yard_stacks = [[] for _ in range(num_stacks)]
    for c in containers:
        yard_stacks[random.randint(0, num_stacks - 1)].append(c)

    # Optionally shuffle stack order (so not all are increasing by id)
    for st in yard_stacks:
        random.shuffle(st)

    return ContainerEnv(yard_stacks, imbalance_bound=imbalance_bound)




if __name__ == "__main__":

    env = random_instance(num_containers=12, num_stacks=3, imbalance_bound=500, max_weight=1)

    print_state("Initial random state", env.initial, env)


    print("A* Search with admissible heuristic:")
    sol, visited_count = astar_search(env)
    print(f"Length of visited: {visited_count}")
    print_state("Final state (A* Search)", sol, env)
    print(reconstruct_plan(sol))
    print(f"Relocation cost: {env.relocation_cost(sol.left)+env.relocation_cost(sol.right)}")
    print("--------------------------------")
    
    sol, visited_count = beam_search(env, beam_width=2)
    print(f"Length of visited: {visited_count}")
    print_state("Final state (Beam Search)", sol, env)
    print(reconstruct_plan(sol))
    print(f"Relocation cost: {env.relocation_cost(sol.left)+env.relocation_cost(sol.right)}")


    print("A* Search with non-admissible heuristic:")
    env.heuristic_type = "non_admissible"
    sol, visited_count = astar_search(env)
    print(f"Length of visited: {visited_count}")
    print_state("Final state (A* Search)", sol, env)
    print(reconstruct_plan(sol))
    print(f"Relocation cost: {env.relocation_cost(sol.left)+env.relocation_cost(sol.right)}")
    print("--------------------------------")
    
    sol, visited_count = beam_search(env, beam_width=2)
    print(f"Length of visited: {visited_count}")
    print_state("Final state (Beam Search with non-admissible heuristic)", sol, env)
    print(reconstruct_plan(sol))
    print(f"Relocation cost: {env.relocation_cost(sol.left)+env.relocation_cost(sol.right)}")
