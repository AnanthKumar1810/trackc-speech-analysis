import unittest
from pathlib import Path
from collections import defaultdict
import numpy as np
from core import loadl, read
from localizer import detect, _merge, _rubric, gate_single_edit_events

ROOT=Path(__file__).resolve().parents[1]

class LocalizerTests(unittest.TestCase):
    def test_single_edit_gate_keeps_only_predicted_type(self):
        events=[{'type':'pause'},{'type':'rush'},{'type':'quiet'}]
        self.assertEqual(gate_single_edit_events(events,'rush'),[events[1]])
        self.assertEqual(gate_single_edit_events(events,'none'),[])

    @classmethod
    def setUpClass(cls):
        cls.has_audio=(ROOT/'audio').is_dir()
        if not cls.has_audio:
            cls.manifest={};cls.words=defaultdict(list)
            return
        cls.manifest={r['clip_id']:r for r in loadl(ROOT/'metadata/manifest.jsonl')}
        cls.words=defaultdict(list)
        for r in loadl(ROOT/'alignment_mfa/word_alignment_mfa.jsonl'):cls.words[r['clip_id']].append(r)

    def clip(self,clip):
        m=self.manifest[clip]
        return read(ROOT/m['audio_path']),dict(np.load(ROOT/m['feature_path']))

    def test_identical_audio_does_not_create_flaw(self):
        if not self.has_audio:self.skipTest('Requires the separately distributed public audio package.')
        x,f=self.clip('reagan_001_base');w=self.words['reagan_001_base']
        r=detect(x,x,w,w,same_setup=True,reference_features=f,take_features=f)
        self.assertEqual(r['events'],[])
        self.assertEqual(r['scores']['combined']['score'],10)

    def test_added_pause_has_detected_interval(self):
        if not self.has_audio:self.skipTest('Requires the separately distributed public audio package.')
        b,bf=self.clip('reagan_001_base')
        u,uf=self.clip('reagan_001_pause_severe')
        r=detect(b,u,self.words['reagan_001_base'],self.words['reagan_001_pause_severe'],
                 same_setup=True,reference_features=bf,take_features=uf)
        self.assertTrue(any(e['type']=='pause' and e['start_seconds']<5.77
                            and e['end_seconds']>4.27 for e in r['events']))

    def test_mismatch_abstains(self):
        if not self.has_audio:self.skipTest('Requires the separately distributed public audio package.')
        x,f=self.clip('reagan_001_base');w=self.words['reagan_001_base']
        r=detect(x,x,w,w[:-1],reference_features=f,take_features=f)
        self.assertEqual(r['status'],'alignment_mismatch')
        self.assertIsNone(r['scores'])

    def test_rubric_charges_pause_by_excess_seconds(self):
        rows=[{'usable':True} for _ in range(12)]
        event={'type':'pause','start_seconds':2.,'end_seconds':3.5,
               'word_index':4,'measurement':1.5,'severity':'severe'}
        score=_rubric(_merge([event]),rows,False)['pauses']
        self.assertEqual(score['score'],5.5)
        self.assertEqual(score['excess_pause_seconds'],1.5)

    def test_rubric_counts_merged_pacing_words_once(self):
        rows=[{'usable':True} for _ in range(10)]
        events=[{'type':'drag','start_seconds':1.,'end_seconds':1.3,
                 'word_index':2,'measurement':1.4,'severity':'severe','explanation':'a'},
                {'type':'drag','start_seconds':1.3,'end_seconds':1.6,
                 'word_index':3,'measurement':1.4,'severity':'severe','explanation':'b'}]
        merged=_merge(events)
        score=_rubric(merged,rows,False)['pacing']
        self.assertEqual(len(merged),1)
        self.assertEqual(score['affected_words'],2)
        self.assertEqual(score['score'],0.)

    def test_configurable_merge_gap_joins_one_edited_region(self):
        events=[{'type':'drag','start_seconds':1.,'end_seconds':1.2,
                 'word_index':2,'measurement':1.4,'severity':'moderate','explanation':'a'},
                {'type':'drag','start_seconds':1.5,'end_seconds':1.8,
                 'word_index':4,'measurement':1.5,'severity':'severe','explanation':'b'}]
        self.assertEqual(len(_merge(events, max_gap=.2)),2)
        merged=_merge(events, max_gap=.4)
        self.assertEqual(len(merged),1)
        self.assertEqual((merged[0]['start_seconds'],merged[0]['end_seconds']),(1.,1.8))
        self.assertEqual(merged[0]['severity'],'severe')

    def test_global_tempo_change_is_not_mislabeled_as_local_word_flaw(self):
        from core import SR
        audio=np.zeros(SR*3,dtype=np.int16)
        ref=[{'word':w,'start_sample':a,'end_sample':b} for w,a,b in
             [('one',1600,4800),('two',8000,11200),('three',14400,17600),('four',20800,24000)]]
        take=[dict(w,start_sample=round(w['start_sample']*.8),
                   end_sample=round(w['end_sample']*.8)) for w in ref]
        f={'time_seconds':np.arange(150)*.02,'rms_dbfs':np.full(150,-30.),
           'relative_pitch_semitones':np.full(150,np.nan)}
        r=detect(audio,audio,ref,take,reference_features=f,take_features=f)
        self.assertEqual(r['normalization']['global_tempo_ratio'],.8)
        self.assertFalse(any(e['type'] in ('rush','drag') for e in r['events']))

    def test_global_recording_gain_is_removed_from_local_level_comparison(self):
        from core import SR
        audio=np.zeros(SR*3,dtype=np.int16)
        words=[{'word':w,'start_sample':a,'end_sample':b} for w,a,b in
               [('one',1600,4800),('two',8000,11200),('three',14400,17600),('four',20800,24000)]]
        f={'time_seconds':np.arange(150)*.02,'rms_dbfs':np.full(150,-30.),
           'relative_pitch_semitones':np.full(150,np.nan)}
        louder=dict(f,rms_dbfs=np.full(150,-24.))
        r=detect(audio,audio,words,words,reference_features=f,take_features=louder)
        self.assertEqual(r['normalization']['global_level_shift_db'],6.)
        self.assertFalse(any(e['type']=='quiet' for e in r['events']))

if __name__=='__main__':unittest.main()
