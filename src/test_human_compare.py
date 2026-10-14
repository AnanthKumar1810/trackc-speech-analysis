import unittest
import numpy as np
from human_compare import compare, measure, SR


class HumanCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t=np.arange(SR*2)/SR
        cls.burst=(.2*np.sin(2*np.pi*180*t)).astype('float32')
        cls.signal=np.r_[cls.burst,np.zeros(int(.2*SR)),cls.burst]

    def test_silence_is_not_a_good_take(self):
        r=compare(self.signal,np.zeros(SR),method='energy')
        self.assertEqual(r['status'],'insufficient_speech')
        self.assertNotIn('pace',r)

    def test_padding_does_not_change_pace(self):
        padded=np.r_[np.zeros(SR),self.signal,np.zeros(SR)]
        r=compare(self.signal,padded,True,True,True,method='energy')
        self.assertAlmostEqual(r['pace']['speech_time_ratio'],1,delta=.03)
        self.assertEqual(r['pace']['direction'],'similar')

    def test_gain_does_not_change_pace(self):
        r=compare(self.signal,self.signal*.25,True,True,True,method='energy')
        self.assertAlmostEqual(r['pace']['speech_time_ratio'],1,delta=.01)
        self.assertAlmostEqual(r['volume']['speech_level_change_db'],-12.04,delta=.1)
        self.assertEqual(r['volume']['direction'],'lower')

    def test_pause_does_not_become_slow_articulation(self):
        take=np.r_[self.burst,np.zeros(SR),self.burst]
        r=compare(self.signal,take,True,True,True,method='energy')
        self.assertAlmostEqual(r['pace']['speech_time_ratio'],1,delta=.03)
        self.assertEqual(len(r['pause_comparison']['candidates']),1)
        self.assertAlmostEqual(r['pause_comparison']['candidates'][0]['extra_seconds'],1,delta=.1)

    def test_unconfirmed_comparison_is_gated(self):
        r=compare(self.signal,self.signal*.25,method='energy')
        self.assertEqual(r['volume']['direction'],'recording_setup_unconfirmed')
        self.assertEqual(r['pace']['direction'],'requires_same_words_and_speaker_confirmation')
        self.assertIsNone(r['human_accuracy'])
        self.assertIsNone(r['severity_rating'])

if __name__=='__main__':unittest.main()
