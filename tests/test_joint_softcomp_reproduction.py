"""Regression checks for the portable full README run."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from ..experiments.joint_softcomp import analyze_simple
from ..experiments.joint_softcomp.uncensored import prepare
from ..evaluation import build_evaluation_time_grid


class ReproductionTest(unittest.TestCase):
    def test_reference_chunks_are_pooled(self):
        metric = dict(brier=.2, auc=.6, ipa=.1, mse_true=.01)
        with tempfile.TemporaryDirectory() as tmp:
            for start in (0, 10):
                payload = dict(beta=0, method='reference', m=0, rep_start=start,
                    test_event_fraction=.4, oracle=metric, softcomp_limit={},
                    cox=dict(runs=[metric]*10), null=dict(runs=[metric]*10))
                Path(tmp, f'simple_b0_reference_m0_r{start}.json').write_text(json.dumps(payload))
            output=io.StringIO()
            with patch.object(sys,'argv',['analyze_simple',tmp]), contextlib.redirect_stdout(output):
                analyze_simple.main()
            self.assertIn('| Cox |  | 20 |',output.getvalue())
            self.assertIn('| No covariate |  | 20 |',output.getvalue())

    def test_censored_pair_shares_subjects_and_975_grid(self):
        for case in (2,3):
            a,b=prepare(case,0,False),prepare(case,0,True)
            for key in ('X_train_full','T_train_true','X_test','eval_times'):
                self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)))
            expected=build_evaluation_time_grid(a.Y_test,a.Delta_test,100,97.5)
            self.assertTrue(torch.isin(expected,a.eval_times).all())
            self.assertTrue((b.Delta_train_full>0).all())
            self.assertTrue((b.Delta_test>0).all())
            self.assertEqual((len(a.X_train_fit),len(a.X_val),len(a.X_test)),(4500,500,1000))
