import io
import tarfile

from pitchlens.data.sportsmot import extract_clips, gt_counts, read_gt


def write_clip(root, name="v_abc_c001"):
    d = root / name
    (d / "gt").mkdir(parents=True)
    (d / "img1").mkdir()
    rows = ["1, 1, 10, 20, 30, 60, 1, 1, 1", "1, 2, 50.5, 20, 30, 80, 1, 1, 1", "2,1,12,20,30,62,1,1,1"]
    (d / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    return d


def test_gt_counts(tmp_path):
    d = write_clip(tmp_path)
    assert read_gt(d).height == 3
    c = gt_counts(d, frames=[1, 2, 3])
    assert c["gt"].to_list() == [2, 1, 0]
    assert c["zoom"][0] == 70.0 and c["zoom"][2] is None
    assert c["sequence"][0] == "v_abc_c001"


def test_extract_only_wanted_clips(tmp_path):
    tar_path = tmp_path / "x.tar"
    with tarfile.open(tar_path, "w") as t:
        for name in ["train/v_keep_c001/gt/gt.txt", "train/v_drop_c002/gt/gt.txt", "../evil/v_keep_c001/x.txt"]:
            data = b"1,1,0,0,1,1,1,1,1\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    found = extract_clips(tar_path, {"v_keep_c001"}, tmp_path / "out")
    assert found == ["v_keep_c001"]
    assert (tmp_path / "out" / "v_keep_c001" / "gt" / "gt.txt").exists()
    assert not (tmp_path / "out" / "v_drop_c002").exists()
