//! What an index run does to the local memory's indexes once they exist: a small run folds its rows
//! in as a delta without retraining, those rows are recalled, and a corpus grown well past what the
//! indexes were trained over is retrained. Own test binary, one test at a time: `$FUNES_HOME` is
//! process-global.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use funes::commands::index::run_index;
use funes::memory::{dataset, Memory};
use lance::index::DatasetIndexExt;
use tokio::sync::Mutex;

static HOME: Mutex<()> = Mutex::const_new(());

/// Enough rows to train IVF_PQ; below a few hundred lance builds no vector index.
const BASE_TURNS: usize = 300;

const SESSION: &str = "refresh-session-0001";

/// Append turns `from..to` of one session, each its own chunk, to the turns file under `source`.
fn write_turns(source: &Path, from: usize, to: usize) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(source.join(format!("{SESSION}.funes.jsonl")))
        .unwrap();
    for i in from..to {
        writeln!(
            f,
            "{}",
            turn(i, &format!("turn {i} about parsing transcripts and lancedb indexing"))
        )
        .unwrap();
    }
}

fn turn(i: usize, text: &str) -> String {
    format!(
        r#"{{"format":1,"session_id":"{SESSION}","cwd":"/home/u/dev/demo","turn_uuid":"turn{i}","seq":{i},"ts":"2026-01-01T00:00:00Z","role":"user","blocks":[{{"block_type":"text","text":"{text}"}}],"harness":"claude"}}"#
    )
}

/// Index a fresh memory of [`BASE_TURNS`] turns; returns its home and the source directory.
async fn indexed_base() -> (tempfile::TempDir, tempfile::TempDir) {
    let home = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    std::env::set_var("FUNES_HOME", home.path());
    write_turns(source.path(), 0, BASE_TURNS);
    run_index(source.path(), false, None).await.unwrap();
    (home, source)
}

/// Each index's segments (base first, then its deltas), by uuid.
async fn segments() -> BTreeMap<String, Vec<String>> {
    let ds = dataset::open(&dataset::table_uri(&dataset::local_memory_dir()), Default::default())
        .await
        .expect("the memory exists");
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for idx in ds.load_indices().await.unwrap().iter() {
        out.entry(idx.name.clone()).or_default().push(idx.uuid.to_string());
    }
    out
}

async fn recall(query: &str) -> String {
    funes::commands::recall::recall(Memory::local(), query.to_string(), 3, 30, 30.0, 0, None, None)
        .await
        .unwrap()
}

#[tokio::test]
async fn a_small_run_appends_a_delta_and_keeps_the_trained_index() {
    let _one_at_a_time = HOME.lock().await;
    let (_home, source) = indexed_base().await;
    let before = segments().await;
    assert_eq!(
        before.len(),
        2,
        "the base run builds the text and vector indexes: {before:?}"
    );

    write_turns(source.path(), BASE_TURNS, BASE_TURNS + 3);
    run_index(source.path(), false, None).await.unwrap();

    for (name, after) in segments().await {
        assert_eq!(
            after.len(),
            2,
            "{name}: one delta beside the base, not a rebuild: {after:?}"
        );
        assert_eq!(after[0], before[&name][0], "{name}: the trained base must survive");
    }
}

#[tokio::test]
async fn rows_a_small_run_adds_are_recalled_by_word_and_by_meaning() {
    let _one_at_a_time = HOME.lock().await;
    let (_home, source) = indexed_base().await;
    let added = "the flamingo quartermaster reconciled the zeppelin ledger";
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(source.path().join(format!("{SESSION}.funes.jsonl")))
        .unwrap();
    writeln!(f, "{}", turn(BASE_TURNS, added)).unwrap();
    run_index(source.path(), false, None).await.unwrap();

    // Words the new row holds and no other does: the BM25 leg's to find.
    let by_word = recall("zeppelin quartermaster").await;
    assert!(by_word.contains(added), "not recalled by its words:\n{by_word}");
    // No word in common with it: only the vector leg can find it.
    let by_meaning = recall("pink wading bird keeping the airship's accounts").await;
    assert!(by_meaning.contains(added), "not recalled by meaning:\n{by_meaning}");
}

#[tokio::test]
async fn doubling_the_corpus_retrains_the_indexes() {
    let _one_at_a_time = HOME.lock().await;
    let (_home, source) = indexed_base().await;
    let before = segments().await;

    write_turns(source.path(), BASE_TURNS, 2 * BASE_TURNS + 20);
    run_index(source.path(), false, None).await.unwrap();

    let after = segments().await;
    assert_eq!(after.len(), 2, "{after:?}");
    for (name, segs) in after {
        assert_eq!(segs.len(), 1, "{name}: retrained into one segment: {segs:?}");
        assert_ne!(segs[0], before[&name][0], "{name}: a retrain replaces the base");
    }
}
