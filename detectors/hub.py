from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from detectors.base import DetectorResult
from detectors.registry import DetectorRegistry


class DetectorHub:
    """Execute normal detectors first, then meta detectors."""

    def __init__(self, parallel: bool = True, max_workers: int = 4):
        self.parallel = parallel
        self.max_workers = max_workers

    @property
    def detectors(self):
        return DetectorRegistry.get_all()

    def _run_single(self, detector, image_bytes: bytes, **kwargs) -> DetectorResult | None:
        try:
            if detector.is_meta:
                return detector.detect(image_bytes, **kwargs)
            return detector.detect(image_bytes)
        except Exception as exc:
            print(f"[WARN] Detector {detector.name} failed: {exc}")
            return None

    def run(self, image_bytes: bytes) -> List[DetectorResult]:
        normal = [detector for detector in self.detectors if not detector.is_meta]
        meta = [detector for detector in self.detectors if detector.is_meta]

        phase1_results = []
        if self.parallel and len(normal) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_single, detector, image_bytes): detector
                    for detector in normal
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        phase1_results.append(result)
        else:
            for detector in normal:
                result = self._run_single(detector, image_bytes)
                if result:
                    phase1_results.append(result)

        phase2_results = []
        for detector in meta:
            result = self._run_single(
                detector,
                image_bytes,
                previous_results=phase1_results,
            )
            if result:
                phase2_results.append(result)

        return phase1_results + phase2_results

    def run_as_dict(self, image_bytes: bytes) -> Dict[str, Any]:
        results = self.run(image_bytes)
        return {result.name: result.features for result in results}
