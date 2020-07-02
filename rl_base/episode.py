"""Defines the tasks that an agent should complete in a given environment."""
from abc import abstractmethod
from typing import (
    Dict,
    Any,
    Tuple,
    TypeVar,
    Generic,
    List,
    Optional,
    Callable,
)

EnvType = TypeVar("EnvType")


class Episode(Generic[EnvType]):
    def __init__(
        self, env: EnvType, task_data: Dict[str, Any], max_steps: int, **kwargs
    ) -> None:
        self._env = env
        self.task_data = task_data
        self._num_steps_taken_in_episode = 0
        self.max_steps = max_steps

    @property
    def environment(self) -> EnvType:
        return self._env

    @abstractmethod
    def state_for_agent(self):
        raise NotImplementedError()

    def step(self, action_as_int: int) -> Dict[str, Any]:
        assert not self.is_paused() and not self.is_complete()
        self._increment_num_steps_taken_in_episode()
        return self._step(action_as_int=action_as_int)

    @abstractmethod
    def _step(self, action_as_int: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def reached_max_steps(self) -> bool:
        return self.num_steps_taken_in_episode() >= self.max_steps

    @abstractmethod
    def reached_terminal_state(self) -> bool:
        raise NotImplementedError()

    def is_complete(self) -> bool:
        return self.reached_terminal_state() or self.reached_max_steps()

    def is_paused(self) -> bool:
        return False

    def _increment_num_steps_taken_in_episode(self) -> None:
        self._num_steps_taken_in_episode += 1

    def num_steps_taken_in_episode(self) -> int:
        return self._num_steps_taken_in_episode

    def info(self):
        return {"ep_length": self.num_steps_taken_in_episode()}

    @property
    def available_actions(self) -> Tuple[str, ...]:
        return tuple(a for g in self.available_action_groups for a in g)

    @property
    def available_action_groups(self) -> Tuple[Tuple[str, ...], ...]:
        return self.class_available_action_groups()

    @classmethod
    def class_available_actions(cls, **kwargs) -> Tuple[str, ...]:
        return tuple(a for g in cls.class_available_action_groups(**kwargs) for a in g)

    @classmethod
    @abstractmethod
    def class_available_action_groups(cls, **kwargs) -> Tuple[Tuple[str, ...], ...]:
        raise NotImplementedError()

    @property
    def total_available_actions(self) -> int:
        return len(self.available_actions)

    def index_to_action(self, index: int) -> str:
        assert 0 <= index < self.total_available_actions
        return self.available_actions[index]


class MultiAgentEpisode(Generic[EnvType]):
    def __init__(
        self,
        env: EnvType,
        task_data: Dict[str, Any],
        max_steps: int,
        before_step_function: Optional[Callable] = None,
        after_step_function: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        self._env = env
        self.task_data = task_data
        self._num_steps_taken_in_episode = 0
        self.max_steps = max_steps

        self.before_step_function = before_step_function
        self.after_step_function = after_step_function

    @property
    def environment(self) -> EnvType:
        return self._env

    @abstractmethod
    def states_for_agents(self):
        raise NotImplementedError()

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        assert not self.is_paused() and not self.is_complete()
        step_results = []
        before_info = (
            None
            if self.before_step_function is None
            else self.before_step_function(episode=self)
        )
        for i in range(self._env.num_agents):
            self._increment_num_steps_taken_in_episode()
            step_results.append(self._step(actions_as_ints[i], agent_id=i))
        if self.after_step_function is not None:
            self.after_step_function(
                step_results=step_results, before_info=before_info, episode=self
            )
        return step_results

    @abstractmethod
    def _step(self, action_as_int: int, agent_id: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def reached_max_steps(self) -> bool:
        return self.num_steps_taken_in_episode() >= self.max_steps

    @abstractmethod
    def reached_terminal_state(self) -> bool:
        raise NotImplementedError()

    def is_complete(self) -> bool:
        return self.reached_terminal_state() or self.reached_max_steps()

    def is_paused(self) -> bool:
        return False

    def _increment_num_steps_taken_in_episode(self) -> None:
        self._num_steps_taken_in_episode += 1

    def num_steps_taken_in_episode(self) -> int:
        return self._num_steps_taken_in_episode

    def info(self):
        return {"ep_length": self.num_steps_taken_in_episode()}

    @property
    def available_actions(self) -> Tuple[str, ...]:
        return tuple(a for g in self.available_action_groups for a in g)

    @property
    def available_action_groups(self) -> Tuple[Tuple[str, ...], ...]:
        return self.class_available_action_groups()

    @classmethod
    def class_available_actions(cls, **kwargs) -> Tuple[str, ...]:
        return tuple(a for g in cls.class_available_action_groups(**kwargs) for a in g)

    @classmethod
    @abstractmethod
    def class_available_action_groups(cls, **kwargs) -> Tuple[Tuple[str, ...], ...]:
        raise NotImplementedError()

    @property
    def total_available_actions(self) -> int:
        return len(self.available_actions)

    def index_to_action(self, index: int) -> str:
        assert 0 <= index < self.total_available_actions
        return self.available_actions[index]
