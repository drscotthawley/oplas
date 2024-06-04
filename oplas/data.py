import torch 
import stempeg 
import multiprocessing as mp 
#from tqdm.contrib.concurrent import process_map # doesn't work great w/ jupyter
from tqdm.contrib.concurrent import process_map 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm, trange 
from functools import partial
from torch.utils.data import Dataset
from glob import glob 
import os
import ctypes
import numpy as np


class StemDataset(Dataset):
    """This reads MUSDB18 stems files in .mp4 format. The contents of these are given in MUSDB18 docs:
        0 - The mixture,    <--- note we're not going to use this
        1 - The drums,
        2 - The bass,
        3 - The rest of the accompaniment,
        4 - The vocals.
    """
    def __init__(self,
        subset      = 'train', # 'train' or 'test'
        data_dir    = '/home/shawley/datasets/musdb18-stems', # dir to look for songs
        preload     = True, # load all audio files into memory at init. If False, load on demand
        share_mem   = True,  # share audio data memory between workers
        chunk_size  = 2**18, # size of audio chunks to return
        sample_rate = 44100, # sample rate of audio
        load_frac   = 1.0, # fraction of dataset to load
        debug       = False, # print debug info
        ):
        search_dir = f'{data_dir}/{subset}'
        self.songs_listed = sorted(glob(f'{search_dir}/*.mp4'))
        print(f"{subset}: {len(self.songs_listed)} songs listed.  preload={preload}")
        self.songs = [None]*len(self.songs_listed)  # actual song data loaded

        # automatically adjust chunk_size to sample rate vs 44100
        #if sample_rate != 44100:  chunk_size = int(chunk_size * sample_rate/44100)
        if debug: print("chunk_size = ",chunk_size)
        
        self.subset, self.chunk_size, self.sample_rate, self.debug = subset, chunk_size, sample_rate, debug
        self.share_mem = share_mem
        self.load_count = 0
        self.song_data = None   #  this will be a shared array to store (zero-padded) song audio, persistently shared between all workers
        if preload: self.preload(load_frac=load_frac)

    def load_song(self, idx, debug=False):
        "loads one song file"
        if type(idx) is int:
            song_file = self.songs_listed[idx] 
        elif type(idx) is str: 
            song_file = idx
        else: 
            print("Unsupported datatype = ",type(idx))
        self.load_count += 1 # note this doesn't really work with parallel loading, i.e. when num_workers>0 :-(
        if debug or self.debug: print(f"{self.subset}: Loading {song_file}", flush=True)
        data, sample_rate = stempeg.read_stems(song_file, sample_rate=self.sample_rate)
        data = torch.tensor(data, dtype=torch.float32)
        if debug: print(f"load_song {idx}: {self.songs_listed[idx]}: data.shape = ",data.shape)
        song_dict = {'name': song_file, 'data': data, 'sample_rate': sample_rate, 'length': data.shape[1]}
        return song_dict 

    def group_audio_data(self):
        """creates a a big (shared memory?) array  for audio data (one that's common to all workers)"""
        n_songs, n_stems, n_channels = len(self.songs), 5, 2
        max_len = 0    # we need to find out the longest song (in samples) to fit in the data array
        for song in self.songs:
            if song is not None:
                max_len = max(max_len, song['length'])
        if self.share_mem: 
            print("     Creating shared memory array...")
            shared_array_base = mp.Array(ctypes.c_float,  n_songs * n_stems * max_len * n_channels) 
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(n_songs, n_stems, max_len, n_channels)
            shared_array = np.zeros((n_songs, n_stems, max_len, n_channels))
            self.song_data = torch.from_numpy(shared_array)
        else:  # to compare against non-sharing way of doing things
            self.song_data = torch.zeros((n_songs, n_stems, max_len, n_channels))
        for i, song in enumerate(self.songs):
            self.song_data[i,:,:song['length'],:] = song['data']  # copy the data over to the shared array
        if self.share_mem: # here's the key: now remove the non-shared audio data from memory!
            for song in self.songs:
                song.pop('data')


    def preload(self, num_workers=min(12, os.cpu_count()), load_frac=1.0, debug=False): 
        """Preloads all songs - in parallel. May not be feasible for large datasets."""
        print(f"{self.subset}: Preloading songs...")
        self.songs = []
        max_ = max(1, int(len(self.songs_listed)*load_frac)) 
        if num_workers > 1:  # parallel loading, fast but often hangs
            with mp.Pool(processes=num_workers) as p:  
                with tqdm(total=max_) as pbar:
                    for r in p.imap_unordered(self.load_song, range(0, max_)):
                        self.songs.append(r)
                        pbar.update()
        else:  # sequential is slow but shure
            for i in trange(max_):
                self.songs.append(self.load_song(i))
        # just to be sure... rewrite the song list (ordering) based on what we got back from the read: 
        self.songs_listed = [x['name'] for x in self.songs]  
        self.group_audio_data() 

    def __len__(self):
        return len(self.songs)*100000 # we're going to be grabbing random windows so...keep the party going

    def __getitem__(self, idx, debug=False):
        """ Returns a random chunk of audio / grouped-stems from a random song"""
        idx = torch.randint(0, len(self.songs), (1,)).item()  # ignore the input idx, pick a random song
        data = self.song_data[idx] #self.songs[idx]['data']
        T = self.songs[idx]['length']  # the real length of the song
        if T < self.chunk_size:
            # we're about to get an error if this is ever true. don't pad with zeros just let it fail
            if debug: print(f"\n__getitem__: songs[{idx}] = ({self.songs_listed[idx]}),  data.shape ={data.shape}, chunk_size = {self.chunk_size}", flush=True)
        start = torch.randint(0, T - self.chunk_size, (1,))
        end = start + self.chunk_size
        out = data[:, start:end, :]
        if debug: print("\n__getitem__: out.shape = ",out.shape)
        return out.to(torch.float32) # .to just to make sure...




class EncodingsDataset(Dataset):
    """This reads precomputed encodings from disk. 
        The encodings are assumed to be in the same order for each part.
    """
    def __init__(self,
        subset     = 'train', # 'train' or 'test'
        data_dir   = '/data/05-03_VGGish_1min_Encodings', # dir to look for songs
        preload    = True, # load all audio files into memory at init. If False, load on demand
        chunk_size = 590, # size of windows of encoding-spectrograms chunks to return
        debug      = False, # print debug info
        ext        = '.pt'
        ):
        # check if '/train' and '/test' dirs exist in data_dir
        if not os.path.isdir(f'{data_dir}/train') or not os.path.isdir(f'{data_dir}/test'):
            print("Taking a moment to build train/ and test/ in data_dir...")
            build_vggish_stemlike(data_dir)

        search_dir = f'{data_dir}/{subset}'
        print("Searching in",search_dir)
        self.songs_listed = sorted(glob(f'{search_dir}/*{ext}'))
        print(f"{subset}: {len(self.songs_listed)} songs listed.  preload={preload}")
        self.songs = [None]*len(self.songs_listed)  # actual song data loaded
        self.subset, self.chunk_size, self.debug = subset, chunk_size, debug
        self.load_count = 0
        if preload: self.preload()

    def load_song(self, idx, debug=False):
        "appends song data self.songs"
        if type(idx) is int:
            song_file = self.songs_listed[idx] 
        elif type(idx) is str: 
            song_file = idx
        else: 
            print("Unsupported datatype = ",type(idx))
        self.load_count += 1 # note this doesn't really work with parallel loading, i.e. when num_workers>0 :-(
        if debug or self.debug: print(f"{self.subset}: Loading {song_file}", flush=True)
        data, sample_rate = torch.load(song_file), 44100
        #data = torch.tensor(data, dtype=torch.float32)
        song_dict = {'name': song_file, 'data': data, 'sample_rate': sample_rate}
        #self.songs[idx] = song_dict  # not great for parallel loading
        return song_dict 

    def preload(self, debug=False): 
        """Preloads all songs - in parallel. May not be feasible for large datasets."""
        print(f"{self.subset}: Preloading songs...")
        self.songs = []
        max_ = len(self.songs_listed)
        with mp.Pool(processes=mp.cpu_count()//8) as p:  # the //8 is just so we get to see the prog bar doing something! ;-) 
            with tqdm(total=max_) as pbar:
                for r in p.imap_unordered(self.load_song, range(0, max_)):
                    self.songs.append(r)
                    pbar.update()
        # just to be sure... rewrite the song list (ordering) based on what we got back from the parallel read: 
        self.songs_listed = [x['name'] for x in self.songs]  # TODO: this should really go the other way

    def __len__(self):
        return len(self.songs)*100000 # we're going to be grabbing random windows so...keep the party going

    def __getitem__(self, idx, debug=False):
        # ignore the input idx, pick a random song
        idx = torch.randint(0, len(self.songs), (1,)).item()
        if debug or self.debug: print(f"idx = {idx}, len(self.songs) {len(self.songs)}")
        if self.songs[idx] is None: self.songs[idx] = self.load_song(idx)
        data = self.songs[idx]['data']
        S, T, C = data.shape  # batch, stems, time, channels
        start = torch.randint(0, T - self.chunk_size, (1,))
        end = start + self.chunk_size
        return data[:, start:end, :]




#--- utility routine: 


def build_vggish_stemlike(data_dir='/data/05-03_VGGish_1min_Encodings'):
    in_subsets = [x+'_VGGish' for x in ['Train','Test']]
    for in_s in in_subsets:
        assert os.path.isdir(f'{data_dir}/{in_s}'), f'{data_dir}/{in_s} does not exist'
    out_subsets = ['train','test']
    """
        0 - The mixture,    
        1 - The drums,
        2 - The bass,
        3 - The rest of the accompaniment,
        4 - The vocals.
    """
    parts = ['mix','drums','bass','other','vocals']
    for out_s in out_subsets:
        os.makedirs(f'{data_dir}/{out_s}', exist_ok=True)
    for in_subset in in_subsets:
        # get a list of input mix files 
        search_str = f"{data_dir}/{in_subset}/{parts[0]}/*.pt"
        in_mix_files = glob(search_str)
        print(f"Searching in {search_str} found {len(in_mix_files)} songs")
        for in_mix in in_mix_files:
            print("in_mix =",in_mix)
            mix = torch.tensor(torch.load(in_mix))
            encoding_stems = torch.empty((5,mix.shape[0],mix.shape[1]))
            encoding_stems[0] = mix
            for i, p in enumerate(parts[1:]):
                in_file = in_mix.replace('mix',p)
                in_stem = torch.tensor(torch.load(in_file))
                encoding_stems[1+i] = in_stem
            out_subset = 'train' if 'train' in in_subset.lower() else 'test'
            out_file = in_mix.replace('/mix','').replace(in_subset,out_subset)
            print("    Saving to",out_file)
            torch.save(encoding_stems,out_file)

        #print(in_files)
    


if __name__ == '__main__':
    import sys 

    # only need to run the following once:     
    build_vggish_stemlike()
    sys.exit(0)

    # test the dataset
    test_ds = EncodingsDataset(subset='test',  preload=True, debug=False)
    ds_iter = iter(test_ds)
    songs = []
    songs.append(next(ds_iter))
    songs.append(next(ds_iter))
    for s, song in enumerate(songs):
        print(f"songs[{s}].shape = ",song.shape)
        for i in range(song.shape[0]):
            for j in range(song.shape[1]):
                vec = song[i,j,:]
                vec_norm = vec.norm()
                print(f"song[{s}][{i},{j},:].norm() = {vec_norm:.3f}")
