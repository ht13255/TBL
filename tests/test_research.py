import json

import numpy as np
import pytest

import tbl


def research_twin():
    packet = tbl.Wavepacket(
        temporal_width=18e-12,
        wavelength=1549.5e-9,
        polarization=(1, 1j),
        purity=0.97,
        chirp=0.2,
        label="QD-A",
    )
    source = tbl.SinglePhotonSource(
        repetition_rate=10e6,
        p_single=0.8,
        p_double=0.02,
        wavepacket=packet,
        emission_jitter=2e-12,
    )
    detectors = tbl.DetectorArray(
        {0: tbl.SNSPD(efficiency=0.85, jitter=3e-12, dead_time=20e-9, channel=4)}
    )
    return tbl.DigitalTwin(source, [tbl.DelayLine(4e-9), tbl.PhaseShifter(0.3)], detectors)


def test_run_manifest_captures_reproducible_configuration_and_environment():
    twin = research_twin()
    first = twin.run(40, seed=2026, integrity=True)
    second = twin.run(40, seed=2026, integrity=True)
    manifest = first.manifest
    assert manifest is not None
    assert manifest.schema == "tbl.run-manifest/1"
    assert manifest.tbl_version == tbl.__version__
    assert manifest.rng_algorithm == "PCG64"
    assert manifest.rng_stream_policy == "seedsequence-hierarchy-v2"
    assert manifest.rng_entropy == 2026
    assert manifest.result_hash_policy == "event-binary-v2"
    assert manifest.seed == 2026
    assert manifest.shots == 40
    assert manifest.units["time"] == "s"
    assert manifest.dependencies["numpy"] == np.__version__
    assert manifest.portable_configuration
    assert manifest.configuration_sha256 == second.manifest.configuration_sha256
    assert manifest.result_sha256 == second.manifest.result_sha256
    assert json.loads(manifest.to_json())["configuration_sha256"] == (manifest.configuration_sha256)


def test_research_bundle_round_trip_is_lossless_and_pickle_free(tmp_path):
    result = research_twin().run(25, seed=17)
    sealed = result.seal()
    bundle = result.save_bundle(tmp_path / "run.npz")
    with np.load(bundle, allow_pickle=False) as archive:
        assert archive["bundle_schema"].item() == "tbl.simulation-result/2"
        assert not any(array.dtype == object for array in archive.values())
    loaded = tbl.load_simulation_result(bundle)
    assert loaded.events == result.events
    assert loaded.time_tags == result.time_tags
    assert loaded.shots == result.shots
    assert loaded.seed == result.seed
    assert loaded.manifest == sealed.manifest
    assert loaded.arrival_times == pytest.approx(result.arrival_times)
    assert loaded.verify_integrity()


def test_research_bundle_round_trips_exponential_wavepacket_profile(tmp_path):
    source = tbl.SinglePhotonSource(
        repetition_rate=1e6,
        p_single=1,
        wavepacket=tbl.Wavepacket.exponential(200e-12, label="radiative"),
    )
    detector = tbl.DetectorArray({0: tbl.SNSPD(efficiency=1, jitter=0, dead_time=0)})
    result = tbl.DigitalTwin(source, [], detector).run(5, seed=2)
    loaded = tbl.load_simulation_result(result.save_bundle(tmp_path / "exponential.npz"))
    assert all(event.photon.wavepacket.profile == "exponential" for event in loaded.events)
    assert loaded.events == result.events
    assert loaded.verify_integrity()


def test_schema_one_gaussian_bundle_remains_readable(tmp_path):
    result = research_twin().run(8, seed=99)
    current = result.save_bundle(tmp_path / "current.npz")
    with np.load(current, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays.pop("wavepacket_profile")
    arrays["bundle_schema"] = np.asarray("tbl.simulation-result/1")
    manifest = json.loads(str(arrays["manifest_json"].item()))
    manifest["result_hash_policy"] = "event-binary-v1"
    manifest["result_sha256"] = tbl.result_sha256(
        result.events, result.time_tags, policy="event-binary-v1"
    )
    arrays["manifest_json"] = np.asarray(json.dumps(manifest))
    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(legacy, **arrays)
    loaded = tbl.load_simulation_result(legacy)
    assert all(event.photon.wavepacket.profile == "gaussian" for event in loaded.events)
    assert loaded.verify_integrity()


def test_research_bundle_detects_modified_payload(tmp_path):
    original = research_twin().run(10, seed=8).save_bundle(tmp_path / "original.npz")
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["event_time"][0] += 1e-9
    modified = tmp_path / "modified.npz"
    np.savez_compressed(modified, **arrays)
    with pytest.raises(tbl.SimulationError, match="checksum mismatch"):
        tbl.load_simulation_result(modified)


def test_research_bundle_rejects_missing_or_misaligned_arrays(tmp_path):
    original = research_twin().run(10, seed=12).save_bundle(tmp_path / "original.npz")
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays.pop("event_mode")
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, **arrays)
    with pytest.raises(tbl.ValidationError, match="missing arrays"):
        tbl.load_simulation_result(missing)

    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["event_mode"] = arrays["event_mode"][:-1]
    misaligned = tmp_path / "misaligned.npz"
    np.savez_compressed(misaligned, **arrays)
    with pytest.raises(tbl.ValidationError, match="invalid shape"):
        tbl.load_simulation_result(misaligned)


def test_callable_configuration_is_explicitly_marked_nonportable():
    source = lambda shot, rng: []  # noqa: E731
    detector = tbl.DetectorArray({0: tbl.SNSPD()})
    result = tbl.DigitalTwin(source, [], detector).run(2, seed=1)
    assert result.manifest is not None
    assert not result.manifest.portable_configuration


def test_unseeded_run_records_entropy_that_reproduces_payload():
    twin = research_twin()
    unseeded = twin.run(30, integrity=True)
    assert unseeded.manifest is not None
    replay = twin.run(30, seed=unseeded.manifest.rng_entropy, integrity=True)
    assert replay.time_tags == unseeded.time_tags
    assert replay.events == unseeded.events
    assert replay.manifest.result_sha256 == unseeded.manifest.result_sha256


def test_simulation_result_reports_exact_detection_probability_interval():
    result = research_twin().run(100, seed=44)
    estimate = result.detection_probability(channel=4)
    observed = len({tag.shot for tag in result.time_tags if tag.channel == 4})
    assert estimate.successes == observed
    assert estimate.trials == 100
    assert estimate.probability == pytest.approx(observed / 100)
    assert estimate.interval[0] <= estimate.probability <= estimate.interval[1]


def test_simulation_result_exposes_autocorrelation_aware_detection_convergence():
    source = tbl.CorrelatedPhotonSource(
        repetition_rate=10e6,
        mean_photon_number=0.7,
        blink_on_to_off=0.01,
        blink_off_to_on=0.02,
    )
    detector = tbl.DetectorArray({0: tbl.SNSPD(efficiency=1, jitter=0, dead_time=0, channel=0)})
    result = tbl.DigitalTwin(source, [], detector).run(20_000, seed=404)
    estimate = result.detection_convergence(channel=0, max_lag=2000)
    assert estimate.integrated_autocorrelation_time > 5
    assert estimate.effective_sample_size < result.shots / 5
    assert len(estimate.batch_estimates) == 20


def test_dataframe_metadata_cannot_override_core_time_tag_columns():
    tag = tbl.TimeTag(
        2e-9,
        3,
        shot=4,
        metadata={"time": 999.0, "channel": 999, "temperature_K": 4.2},
    )
    frame = tbl.SimulationResult((), (tag,), 5, seed=1).tags_dataframe()
    assert frame.loc[0, "time"] == pytest.approx(2e-9)
    assert frame.loc[0, "channel"] == 3
    assert frame.loc[0, "temperature_K"] == pytest.approx(4.2)


def test_rng_stages_isolate_detector_results_from_unrelated_component_draws():
    class ConsumeRandomNumbers:
        def process(self, events, rng, context):
            del context
            rng.random(10_000)
            return [event.copy() for event in events]

    source = tbl.SinglePhotonSource(repetition_rate=10e6, p_single=1)
    detector = tbl.DetectorArray({0: tbl.SNSPD(efficiency=0.45, jitter=0, dead_time=0)})
    reference = tbl.DigitalTwin(source, [], detector).run(1000, seed=91)
    consuming = tbl.DigitalTwin(source, [ConsumeRandomNumbers()], detector).run(1000, seed=91)
    assert consuming.time_tags == reference.time_tags


@pytest.mark.parametrize("seed", [-1, 1.2, True])
def test_digital_twin_rejects_invalid_seed(seed):
    with pytest.raises(tbl.ValidationError, match="seed"):
        research_twin().run(1, seed=seed)


def test_bundle_requires_versioned_manifest_and_npz_extension(tmp_path):
    result = tbl.SimulationResult((), (), 1, 3)
    with pytest.raises(tbl.ValidationError, match=r"\.npz extension"):
        result.save_bundle(tmp_path / "result.dat")
    with pytest.raises(tbl.ValidationError, match="without a run manifest"):
        result.save_bundle(tmp_path / "result.npz")


def test_in_memory_result_and_manifest_mutation_are_detected(tmp_path):
    result = research_twin().run(10, seed=3, integrity=True)
    result.events[0].metadata["modified"] = True
    with pytest.raises(tbl.SimulationError, match="checksum mismatch"):
        result.verify_integrity()
    with pytest.raises(tbl.SimulationError, match="modified after the run"):
        result.save_bundle(tmp_path / "modified.npz")

    manifest_result = research_twin().run(5, seed=4)
    manifest_result.manifest.configuration["changed"] = True
    with pytest.raises(tbl.SimulationError, match="provenance was modified"):
        manifest_result.verify_integrity()


def test_unsealed_result_can_be_sealed_without_mutating_original():
    result = research_twin().run(5, seed=14)
    assert result.manifest.result_sha256 is None
    with pytest.raises(tbl.ValidationError, match="not sealed"):
        result.verify_integrity()
    sealed = result.seal()
    assert sealed is not result
    assert sealed.manifest.result_sha256 is not None
    assert result.manifest.result_sha256 is None
    assert sealed.verify_integrity()


@pytest.mark.parametrize("value", [0.0, -1e-9, np.nan, np.inf])
def test_digital_twin_rejects_invalid_time_bin_width(value):
    detector = tbl.DetectorArray({0: tbl.SNSPD()})
    with pytest.raises(tbl.ValidationError, match="time_bin_width"):
        tbl.DigitalTwin(lambda shot, rng: [], [], detector, time_bin_width=value)


def test_digital_twin_rejects_nonfinite_or_reversed_acquisition_windows():
    twin = research_twin()
    with pytest.raises(tbl.ValidationError, match="acquisition_start"):
        twin.run(1, acquisition_start=np.nan)
    with pytest.raises(tbl.ValidationError, match="acquisition_end"):
        twin.run(1, acquisition_start=2.0, acquisition_end=1.0)
