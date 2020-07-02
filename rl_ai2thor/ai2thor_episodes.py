from abc import abstractmethod, ABC
from typing import Dict, Any, Optional, Sequence, Tuple, List

from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_base import Episode
from rl_base.episode import MultiAgentEpisode


class AI2ThorEpisode(Episode[AI2ThorEnvironment]):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ) -> None:
        super(AI2ThorEpisode, self).__init__(
            env=env, task_data=task_data, max_steps=max_steps, **kwargs
        )
        self._last_action = None
        self._last_action_success = None

    def last_action(self):
        return self._last_action

    def last_action_success(self):
        return self._last_action_success

    def step(self, action_as_int: int) -> Dict[str, Any]:
        step_result = super(AI2ThorEpisode, self).step(action_as_int=action_as_int)
        self._last_action = action_as_int
        self._last_action_success = self.environment.last_event.metadata[
            "lastActionSuccess"
        ]
        step_result["action"] = self._last_action
        step_result["action_success"] = self._last_action_success
        return step_result

    def state_for_agent(self):
        state = {
            "frame": self.environment.current_frame,
            "last_action": self._last_action,
            "last_action_success": self._last_action_success,
        }
        return state

    @abstractmethod
    def _step(self, action_as_int: int) -> Dict[str, Any]:
        raise NotImplementedError()


class MultiAgentAI2ThorEpisode(MultiAgentEpisode[AI2ThorEnvironment], ABC):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        task_data: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ) -> None:
        super(MultiAgentAI2ThorEpisode, self).__init__(
            env=env, task_data=task_data, max_steps=max_steps, **kwargs
        )
        self._last_actions: Optional[Sequence[int]] = None
        self._last_actions_success: Optional[Sequence[bool]] = None

        self.include_depth_frame = kwargs.get("include_depth_frame", False)

    def last_actions(self):
        return self._last_actions

    def last_actions_success(self):
        return self._last_actions_success

    def multi_step(self, actions_as_ints: Tuple[int, ...]) -> List[Dict[str, Any]]:
        assert not self.is_paused() and not self.is_complete()
        step_results = []
        before_info = (
            None
            if self.before_step_function is None
            else self.before_step_function(episode=self)
        )
        for agent_id in range(self._env.num_agents):
            self._increment_num_steps_taken_in_episode()
            step_result = self._step(actions_as_ints[agent_id], agent_id=agent_id)
            step_result["action"] = actions_as_ints[agent_id]
            step_result["action_success"] = self.environment.last_event.metadata[
                "lastActionSuccess"
            ]
            step_results.append(step_result)
        if self.after_step_function is not None:
            self.after_step_function(
                step_results=step_results, before_info=before_info, episode=self
            )
        return step_results

    def states_for_agents(self):
        states = []
        for agent_id in range(self.environment.num_agents):
            e = self.environment.last_event.events[agent_id]
            last_action = (
                None if self._last_actions is None else self._last_actions[agent_id]
            )
            last_action_success = (
                None
                if self._last_actions_success is None
                else self._last_actions_success[agent_id]
            )
            states.append(
                {
                    "frame": e.frame,
                    "last_action": last_action,
                    "last_action_success": last_action_success,
                }
            )
            if self.include_depth_frame:
                states[-1]["depth_frame"] = e.depth_frame
        return states
