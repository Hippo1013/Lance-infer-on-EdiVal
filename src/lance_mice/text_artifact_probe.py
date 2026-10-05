"""Opt-in prompt ablations; never selected by benchmark inference."""
from .omni_pipeline import LanceHistoryPipeline
from .protocol import LEGACY_PROTOCOL_VERSION, Segment, history_segments


class TextArtifactProbePipeline(LanceHistoryPipeline):
    def forward(self, req):
        prompt = req.prompts[0] if req.prompts else None
        history = (prompt.get('extra_args') or {}).get('lance_history') if isinstance(prompt, dict) else None
        self._text_probe_mode = history.get('text_probe_mode', 'full') if history else 'full'
        return super().forward(req)

    def _history_segments(self, instructions):
        mode = self._text_probe_mode
        if mode not in {'full', 'label_only', 'framing_only', 'bare', 'edit_labels', 'last_image'}:
            raise ValueError(mode)
        segments = history_segments(instructions, version=LEGACY_PROTOCOL_VERSION)
        if mode in {'label_only', 'bare', 'edit_labels', 'last_image'}:
            segments = [s for s in segments if s.role != 'framing']
        if mode in {'framing_only', 'bare', 'last_image'}:
            # Keep a zero-token boundary marker for the existing CFG split.
            segments = [Segment('text', 'label', s.turn, '') if s.role == 'label' else s
                        for s in segments]
        if mode == 'edit_labels':
            segments = [Segment('text', 'label', s.turn, 'Edit: ') if s.role == 'label' else s
                        for s in segments]
        if mode == 'last_image' and len(instructions) > 1:
            # Immediately precede the current instruction; retain in both CFG branches.
            segments = [Segment('text', 'label', s.turn, 'Edit the last image.\n')
                        if s.role == 'label' and s.turn == len(instructions) else s
                        for s in segments]
        return segments
