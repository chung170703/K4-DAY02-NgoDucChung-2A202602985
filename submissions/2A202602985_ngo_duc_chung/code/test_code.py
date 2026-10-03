"""Test CPU cho code của bạn (không cần GPU, không cần tải DeepWeeds: dùng dữ liệu giả nhỏ).

Chạy từ thư mục code/:   python -m unittest test_code -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import benchmark as B
import checks
import dataset as D
import final as FN
import inference as I
import losses as L
import model as M
import tables as TB
import train as T


def make_fake_dataset(root: Path, n=200, size=64, k=9, seed=0):
    rng = np.random.default_rng(seed)
    img_dir, lab_dir = root / "images", root / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir()
    rows = []
    for i in range(n):
        lab = i % k
        arr = np.full((size, size, 3), 20 + 25 * lab, np.uint8)  # màu theo lớp -> học được
        arr = np.clip(arr + rng.integers(-10, 10, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / f"im{i}.jpg")
        rows.append((f"im{i}.jpg", lab, str(lab)))
    df = pd.DataFrame(rows, columns=["Filename", "Label", "Species"]).sample(frac=1, random_state=0)
    a, b = int(0.6 * n), int(0.8 * n)
    for name, part in zip(("train", "val", "test"), (df[:a], df[a:b], df[b:])):
        part.to_csv(lab_dir / f"{name}_subset0.csv", index=False)
    df.to_csv(lab_dir / "labels.csv", index=False)
    return img_dir, lab_dir


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(32, 9)
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        self.assertLess(abs(L.FocalLoss(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(L.FocalLoss(2.0)(self.z, self.y), F.cross_entropy(self.z, self.y))

    def test_label_smoothing0_equals_ce(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)), 1e-6)

    def test_label_smoothing_matches_torch(self):
        a = L.LabelSmoothingCE(0.1)(self.z, self.y)
        b = F.cross_entropy(self.z, self.y, label_smoothing=0.1)
        self.assertLess(abs(a - b), 1e-5)

    def test_class_weights(self):
        counts = [1000, 1000, 100, 100, 1000, 1000, 1000, 1000, 9000]
        w0 = L.class_weights(counts, 0.0)
        self.assertAlmostEqual(w0.mean().item(), 1.0, places=5)
        self.assertGreater(w0[2], w0[8])
        w = L.class_weights(counts, 0.999)
        self.assertAlmostEqual(w.sum().item(), 9.0, places=4)
        self.assertGreater(w[2], w[8])

    def test_cutmix_lambda_matches_area(self):
        np.random.seed(0)
        x = torch.zeros(8, 3, 32, 32)
        xb = torch.ones(8, 3, 32, 32)
        # ảnh nguồn = 0, ảnh dán = 1 -> tỉ lệ pixel = 1 chính là (1 - lam) nếu lam đúng diện tích
        for _ in range(20):
            y = torch.arange(8)
            xm, (ya, yb, lam) = L.mix_batch(torch.where(torch.arange(8).view(8, 1, 1, 1) % 2 == 0, x, xb), y, 1.0, "cutmix")
            self.assertTrue(0.0 <= lam <= 1.0)
        # kiểm tra trực tiếp: tất cả ảnh giống nhau theo từng pixel -> sau trộn vẫn đúng diện tích
        xs = torch.arange(8).float().view(8, 1, 1, 1).expand(8, 3, 32, 32).contiguous()
        xm, (ya, yb, lam) = L.mix_batch(xs, torch.arange(8), 1.0, "cutmix")
        frac_a = (xm[0, 0] == xs[0, 0, 0, 0]).float().mean().item()
        if ya[0] != yb[0]:
            self.assertAlmostEqual(frac_a, lam, places=5)

    def test_mixup_and_mixed_loss(self):
        np.random.seed(1)
        x = torch.randn(8, 3, 16, 16)
        y = torch.randint(0, 9, (8,))
        xm, tg = L.mix_batch(x, y, 0.4, "mixup")
        self.assertEqual(xm.shape, x.shape)
        ce = torch.nn.CrossEntropyLoss()
        z = torch.randn(8, 9)
        ya, yb, lam = tg
        self.assertAlmostEqual(L.mixed_loss(ce, z, tg).item(),
                               (lam * ce(z, ya) + (1 - lam) * ce(z, yb)).item(), places=6)

    def test_build_criterion(self):
        for kind, kw in [("ce", {}), ("ls", {"smoothing": 0.1}), ("focal", {"gamma": 2.0}),
                         ("ce_weighted", {"weight": torch.ones(9)})]:
            self.assertTrue(torch.isfinite(L.build_criterion(kind, **kw)(self.z, self.y)))
        with self.assertRaises(ValueError):
            L.build_criterion("nope")


class TestModel(unittest.TestCase):
    def test_param_groups_three_groups(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        g = M.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual([x["name"] for x in g], ["backbone", "backbone_norm_bias", "head"])
        self.assertEqual(g[1]["weight_decay"], 0.0)
        self.assertEqual(g[2]["lr"], 1e-3)
        n = sum(len(x["params"]) for x in g)
        self.assertEqual(n, len(list(m.parameters())))
        self.assertEqual(m.pretrained_tag, "none")

    def test_freeze_keeps_bn_eval(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        M.freeze_backbone(m)
        M.set_train_mode(m)
        self.assertTrue(all(p.requires_grad for p in m.fc.parameters()))
        self.assertFalse(any(p.requires_grad for n, p in m.named_parameters() if not n.startswith("fc.")))
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(before, m.bn1.running_mean))
        self.assertEqual(len(M.param_groups(m, 1e-4, 1e-3, 0.05)), 1)

    def test_vit_no_decay_params(self):
        m = M.build_model("deit_small_patch16_224", pretrained=False, init="scratch")
        g = M.param_groups(m, 1e-4, 1e-3, 0.05)
        nd = {id(p) for p in g[1]["params"]}
        self.assertIn(id(m.pos_embed), nd)

    def test_counts(self):
        m = M.build_model("resnet50", pretrained=False, init="scratch")
        self.assertAlmostEqual(M.count_params(m), 23.5, delta=0.3)  # 25.6M có head 1000 lớp
        self.assertAlmostEqual(M.count_gmacs(m, 224), 4.1, delta=0.2)


class TestInference(unittest.TestCase):
    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        n, k = 4000, 9
        z = rng.normal(size=(n, k)) * 2
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        y = np.array([rng.choice(k, p=pi) for pi in p])
        T_hat = I.fit_temperature(z * 3.0, y)  # logit bị "quá tự tin" 3 lần -> T ≈ 3
        self.assertAlmostEqual(T_hat, 3.0, delta=0.25)
        probs = I.apply_temperature(z * 3.0, T_hat)
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
        np.testing.assert_array_equal(probs.argmax(1), (z * 3.0).argmax(1))

    def test_aggregate_and_ensemble(self):
        z1, z2 = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views([z1, z2], space).sum(1), 1.0, atol=1e-9)
        self.assertEqual(I.aggregate_views([z1], "prob").shape, (5, 9))
        p = I.ensemble_probs([I.apply_temperature(z1, 1), I.apply_temperature(z2, 1)])
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        with self.assertRaises(ValueError):
            I.aggregate_views([z1], "x")

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        self.assertEqual(len(I.views_multicrop(x, 6)), 5)
        self.assertEqual(len(I.views_multicrop(x, 6, flip=True)), 10)
        self.assertTrue(all(v.shape[-1] == 6 for v in I.views_multicrop(x, 6)))
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [4, 8, 12])], [4, 8, 12])

    def test_fuse_conv_bn(self):
        for name in ("resnet18", "mobilenetv3_small_100", "efficientnet_b0", "resnext50_32x4d"):
            m = M.build_model(name, pretrained=False, init="scratch").eval()
            for mod in m.modules():  # BN mặc định mean 0 var 1 -> không kiểm tra được gì: làm ngẫu nhiên
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.3)
                    mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.data.uniform_(0.5, 1.5)
                    mod.bias.data.normal_(0, 0.3)
            fused = I.fuse_conv_bn(m)
            self.assertGreater(fused.fuse_pairs, 5, name)
            self.assertLess(fused.fuse_max_abs_err, 1e-4, name)
            n_bn = sum(isinstance(x, torch.nn.BatchNorm2d) for x in fused.modules())
            self.assertEqual(n_bn, 0, f"{name}: còn {n_bn} BN chưa gộp")
        vit = M.build_model("deit_small_patch16_224", pretrained=False, init="scratch")
        self.assertEqual(I.fuse_conv_bn(vit).fuse_pairs, 0)


class TestBenchmark(unittest.TestCase):
    def test_latency_cpu(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        r = B.latency_report(m, 1, 64, "fp32", "cpu", warmup=10, iters=50)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        self.assertEqual(r["n"], 50)
        t = B.tta_latency(m, 3, img_size=64, device="cpu", warmup=10, iters=50)
        self.assertTrue(0.5 < t["ratio_vs_k_x_single"] < 1.5)  # K view ≈ K lần 1 view
        with self.assertRaises(ValueError):
            B.bench(lambda: None, warmup=1, iters=5)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.images, cls.labels = make_fake_dataset(cls.root)
        cls._total = D.TOTAL_IMAGES
        D.TOTAL_IMAGES = 200  # dữ liệu giả có 200 ảnh

    @classmethod
    def tearDownClass(cls):
        D.TOTAL_IMAGES = cls._total
        cls.tmp.cleanup()

    def cfg(self, **kw):
        base = dict(exp_id="X", backbone="resnet18", init="scratch", img_size=64, epochs=3, batch_size=16,
                    num_workers=0, amp=False, images_dir=str(self.images), labels_dir=str(self.labels),
                    out_dir=str(self.root / "runs"), pred_dir=str(self.root / "pred"),
                    curves_dir=str(self.root / "curves"), warmup_epochs=0.5, lr_backbone=1e-3, lr_head=1e-2)
        base.update(kw)
        return T.Config(**base)

    def test_check_split(self):
        tr, va, te = D.load_split(self.labels)
        info = D.check_split(tr, va, te, self.images)
        self.assertEqual(sum(info["n"].values()), 200)
        with self.assertRaises(AssertionError):
            D.check_split(tr, tr, te, self.images)

    def test_loaders(self):
        tr, va, _ = D.load_split(self.labels)
        lo = D.make_loader(va, self.images, D.build_transforms(False, 64), 16, False, num_workers=0)
        names = [f for _, _, fs in lo for f in fs]
        self.assertEqual(names, va["Filename"].tolist())  # thứ tự ổn định
        x, y, f = next(iter(D.make_loader(tr, self.images, D.build_transforms(True, 64, "geo"), 16, True,
                                          "balanced", 0)))
        self.assertEqual(tuple(x.shape), (16, 3, 64, 64))
        for aug in D.AUG_CHOICES:
            self.assertEqual(tuple(D.build_transforms(True, 32, aug)(Image.new("RGB", (64, 64))).shape), (3, 32, 32))
        ds = D.DeepWeedsDataset(va.head(5), self.images, D.build_transforms(False, 64), cache=True)
        self.assertEqual(ds[0][2], va.iloc[0]["Filename"])

    def test_scheduler_shape(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        cfg = self.cfg(epochs=10, warmup_epochs=1.0)
        opt = T.build_optimizer(m, cfg)
        sch = T.build_scheduler(opt, cfg, 20)
        lrs = []
        for _ in range(200):
            lrs.append(opt.param_groups[-1]["lr"])
            opt.step()
            sch.step()
        self.assertLess(lrs[0], lrs[10] if len(lrs) > 10 else 1)
        self.assertAlmostEqual(max(lrs), cfg.lr_head, delta=cfg.lr_head * 0.06)
        self.assertEqual(int(np.argmax(lrs)), 19)       # đỉnh ở cuối warmup
        self.assertLess(lrs[-1], cfg.lr_head * 0.01)    # cosine về gần 0

    def test_ema(self):
        m = M.build_model("resnet18", pretrained=False, init="scratch")
        ema = T.EMA(m, 0.9)
        w0 = ema.module.fc.weight.clone()
        with torch.no_grad():
            m.fc.weight.add_(1.0)
        ema.update(m)
        self.assertFalse(torch.equal(w0, ema.module.fc.weight))
        self.assertTrue(torch.all(ema.module.fc.weight < m.fc.weight))
        self.assertFalse(ema.module.training)

    def test_parse_overrides(self):
        o = T.parse_overrides(["seed=2", "loss=focal", "ema_decay=none", "amp=false", "lr_head=1e-2", "mix=cutmix"])
        self.assertEqual(o, {"seed": 2, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 1e-2,
                             "mix": "cutmix"})
        with self.assertRaises(ValueError):
            T.parse_overrides(["nope=1"])

    def test_run_end_to_end_and_eval_contract(self):
        import eval as ev
        cfg = self.cfg(exp_id="E2E", seed=1, save_test_predictions=True, ema_decay=0.9, mix="cutmix",
                       loss="focal", class_weight_beta=0.99, sampler="balanced")
        s = T.run(cfg)
        self.assertGreater(s["val_macro_f1"], 1 / 9)  # hơn ngẫu nhiên; cấu hình này cố tình khó
        rd = T.run_dir(cfg)
        for f in ("config.json", "history.csv", "best.pt", "val_logits.npy", "test_logits.npy", "summary.json"):
            self.assertTrue((rd / f).exists(), f)
        self.assertFalse((rd / "last.pt").exists())
        self.assertTrue(T.curve_path(cfg).exists())
        pred = ev.read_pred(str(T.pred_path(cfg, "test")))
        ev.check_against_csv(pred, str(self.labels / "test_subset0.csv"))
        self.assertAlmostEqual(ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)["macro_f1"],
                               s["test_macro_f1"], places=6)
        self.assertEqual(T.run(cfg), s)  # skip_if_done
        # test KHÔNG được ghi khi không bật cờ
        cfg2 = self.cfg(exp_id="NOTEST", epochs=1)
        T.run(cfg2)
        self.assertFalse(T.pred_path(cfg2, "test").exists())
        self.assertTrue(T.pred_path(cfg2, "val").exists())

    def test_learns_plain_recipe(self):
        s = T.run(self.cfg(exp_id="PLAIN", epochs=6, seed=0))
        self.assertGreater(s["val_macro_f1"], 0.6)  # màu theo lớp -> công thức nền phải học được

    def test_resume_matches_uninterrupted(self):
        full = T.run(self.cfg(exp_id="FULL", epochs=3, seed=3))
        cfg = self.cfg(exp_id="INT", epochs=3, seed=3)
        real_eval, calls = T.evaluate, {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # ngắt khi đang đánh giá epoch 2: last.pt mới có epoch 1
                raise KeyboardInterrupt
            return real_eval(*a, **k)

        T.evaluate = flaky
        try:
            with self.assertRaises(KeyboardInterrupt):
                T.run(cfg)
        finally:
            T.evaluate = real_eval
        self.assertTrue((T.run_dir(cfg) / "last.pt").exists())
        resumed = T.run(cfg)
        h_full = pd.read_csv(T.run_dir(self.cfg(exp_id="FULL", seed=3)) / "history.csv")
        h_res = pd.read_csv(T.run_dir(cfg) / "history.csv")
        self.assertEqual(len(h_res), 3)
        np.testing.assert_allclose(h_res["train_loss"], h_full["train_loss"], atol=1e-4)
        np.testing.assert_allclose(h_res["val_macro_f1"], h_full["val_macro_f1"], atol=1e-6)
        self.assertEqual(resumed["best_epoch"], full["best_epoch"])

    def test_frozen_and_checks(self):
        cfg = self.cfg(exp_id="FRZ", init="frozen", epochs=2, backbone="resnet18")
        s = T.run(cfg)
        self.assertIn("val_macro_f1", s)
        tr = checks.overfit_one_batch(self.cfg(), n=8, steps=40)
        self.assertLess(tr[-1], tr[0])

    def test_load_best_and_predict(self):
        cfg = self.cfg(exp_id="LB", epochs=1)
        T.run(cfg)
        model = T.load_best(cfg, torch.device("cpu"))
        _, va, _ = D.load_split(self.labels)
        lo = D.make_loader(va, self.images, D.build_transforms(False, 64), 16, False, num_workers=0)
        names, y, logits = I.predict_logits(model, lo, torch.device("cpu"))
        saved = np.load(T.run_dir(cfg) / "val_logits.npy")
        np.testing.assert_allclose(logits, saved, atol=1e-4)  # suy luận lại khớp logit lúc huấn luyện
        _, _, flips = I.predict_multiview(model, lo, torch.device("cpu"), [I.view_identity, I.view_hflip])
        self.assertEqual(len(flips), 2)

    def test_temperature_views_single_view_matches(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=(500, 9)) * 3
        y = rng.integers(0, 9, 500)
        self.assertAlmostEqual(I.fit_temperature_views([z], y, "prob"), I.fit_temperature(z, y), places=3)
        self.assertAlmostEqual(I.fit_temperature_views([z], y, "logit"), I.fit_temperature(z, y), places=3)

    def test_final_predict_contract_and_test_once(self):
        import eval as ev
        for tta in ("none", "hflip", "crop5"):
            cfg = self.cfg(exp_id=f"FIN{tta}", epochs=1, seed=0)
            T.run(cfg)  # save_test_predictions=False
            self.assertFalse(T.pred_path(cfg, "test").exists())
            out = FN.final_predict(cfg, tta=tta, space="prob")
            self.assertGreater(out["T"], 0)
            for split in ("val", "test"):
                p = ev.read_pred(str(T.pred_path(cfg, split)))
                ev.check_against_csv(p, str(self.labels / f"{split}_subset0.csv"), split)
            u = ev.read_pred(str(T.pred_path(T.Config(**{**cfg.__dict__, "exp_id": cfg.exp_id + "uncal"}), "test")))
            c = ev.read_pred(str(T.pred_path(cfg, "test")))
            np.testing.assert_array_equal(u.y_pred, c.y_pred) if tta == "none" else None  # T không đổi argmax (1 view)
            with self.assertRaises(FileExistsError):
                FN.final_predict(cfg, tta=tta)  # không được chạy test lần hai

    def test_tables(self):
        T.run(self.cfg(exp_id="TB1", epochs=1, seed=0))
        T.run(self.cfg(exp_id="TB1", epochs=1, seed=1))
        df = TB.collect_summaries(self.root / "runs", "TB1")
        self.assertEqual(len(df), 2)
        ms = TB.mean_std_table(df)
        self.assertEqual(ms.loc[0, "n_seeds"], 2)
        self.assertAlmostEqual(ms.loc[0, "val_macro_f1_std"], df["val_macro_f1"].std(ddof=1), places=9)
        path = TB.write_results_xlsx(self.root / "r.xlsx", {"Backbones": df, "Summary": ms}, {"Backbones": "val_macro_f1"})
        back = pd.read_excel(path, sheet_name=None)
        self.assertEqual(set(back), {"Backbones", "Summary"})


if __name__ == "__main__":
    unittest.main()
