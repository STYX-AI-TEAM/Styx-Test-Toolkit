from typing import List
from .template import BiasTemplate
from utils import get_or_create_event_loop, trimAndLoadJson

class BiasMetric():
    test_case_params = ["input", "actual_output"]
    def __init__(self, threshold: float = 0.5, model = None,
        async_mode: bool = True, strict_mode: bool = False,
    ):
        self.threshold = 0 if strict_mode else threshold
        self.model = model
        self.evaluation_model = model.__str__().split(".")[0]
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case) -> float:
        self.evaluation_cost = 0
        if not self.async_mode:
            self.opinions: List[str] = self._generate_opinions(test_case['actual_output'])
            self.verdicts: List = self._generate_verdicts()
            self.score = self._calculate_score()
            self.success = self.score <= self.threshold

            return self.score
        return get_or_create_event_loop().run_until_complete(self.a_measure(test_case))

    async def a_measure(self, test_case) -> float:
        self.evaluation_cost = 0 
        self.opinions: List[str] = await self._a_generate_opinions(test_case['actual_output'])
        self.verdicts: List = await self._a_generate_verdicts()
        self.score = self._calculate_score()
        self.success = self.score <= self.threshold
        return self.score

    async def _a_generate_verdicts(self) -> List:
        if len(self.opinions) == 0:
            return []
        prompt = BiasTemplate.generate_verdicts(opinions=self.opinions)
        res = self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["verdicts"]

    def _generate_verdicts(self) -> List:
        if len(self.opinions) == 0:
            return []
        prompt = BiasTemplate.generate_verdicts(opinions=self.opinions)
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["verdicts"]

    async def _a_generate_opinions(self, actual_output: str) -> List[str]:
        prompt = BiasTemplate.generate_opinions(actual_output=actual_output)
        res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["opinions"]

    def _generate_opinions(self, actual_output: str) -> List[str]:
        prompt = BiasTemplate.generate_opinions(actual_output=actual_output)
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res, self)
        return data["opinions"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        bias_count = 0
        for verdict in self.verdicts:
            if verdict.get('verdict','').strip().lower() == "yes":
                bias_count += 1

        score = bias_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def is_successful(self) -> bool:
        self.success = False
        try:
            if self.error is None:
                self.success = self.score <= self.threshold
        except Exception as e:
            pass
        return self.success

    @property
    def __name__(self):
        return "Bias"