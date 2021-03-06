'''
Created on May 15, 2013

@author: gerdogan
'''
import numpy as np
from aomr_simple_grammar import *
from aomr_grammar import *
from rational_rules import *
import cPickle as pickle
import time
from copy import copy

class MCMCSample:
    def __init__(self, index, state, posterior, move_name):
        self.index = index
        self.state = state
        self.posterior = posterior
        self.move_name = move_name
        
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        
        return self.state == other.state
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __repr__(self):
        return 'Sample {0:d}: {1:s} ({2:G})-({3:s})'.format(self.index, self.state, self.posterior, self.move_name)
    def __str__(self):
        return 'Sample {0:d}: {1:s} ({2:G})-({3:s})'.format(self.index, self.state, self.posterior, self.move_name)
    
class MCMCRunResults:
    """
    Class for storing MCMCSampler results
    Holds info about the run and the samples for each run in ``samples`` list
    """
    def __init__(self, info, state_class, run_params):
        self.info = info
        self.state_class = state_class
        self.run_params = run_params
        self.samples = [[] for i in range(self.run_params['runs'])]
        self.best_samples = [[] for i in range(self.run_params['runs'])]
        self.initial_states = [[] for i in range(self.run_params['runs'])]
        self.acceptance_rate = [[] for i in range(self.run_params['runs'])]
        self.start_time = time.localtime()
        self.end_time = time.localtime()
    
    def save(self):
        """
        Saves results to disk using pickle
        """
        fname = self.run_params['results_folder'] + self.info.replace(' ', '_') + \
                time.strftime('%Y%m%d%H%M%S', self.start_time) + '.mcmc'
        f = open(fname, 'wb')
        pickle.dump(self, f)
        
    def load(self, filename):
        f = open(filename, 'rb')
        results = pickle.load(f)
        return results

    def __str__(self):
        header = self.__repr__()
        body = ''
        for i in range(self.run_params['runs']):
            body = '{0} Run {1}: (Acceptance Rate: {2:f}\n'.format(body, i, self.acceptance_rate[i])
            body = body + '\tSamples\n'
            for sample in self.samples[i]:
                body = body + '\t\t' + repr(sample) + '\n'
                  
            body = body + '\tBest Samples\n'
            for sample in self.best_samples[i]:
                body = body + '\t\t' + repr(sample) + '\n'
        
        return header + '\n' + body
    
    def __repr__(self):
        s = 'MCMC Run Results for ' + self.info + '\n' + \
            'State class: ' + repr(self.state_class) + '\n' + \
            'Time (start-end):' + time.strftime('%Y%m%d%H%M%S', self.start_time) + '-' + time.strftime('%Y%m%d%H%M%S', self.end_time) + '\n' + \
            'Run parameters: ' + repr(self.run_params)
        return s
        

class MCMCSampler:
    """
    Runs MCMC sampler and returns samples
    """
    def __init__(self, sampler_params, initial_state):
        """
        sampler_params: dictionary of sampler parameters
            runs: number of chains run
            iters: number of iterations in one chain
            keep_top_n: number of best samples to keep for one chain
            verbose: if True prints progress info as sampling goes on
            burn_in: number of iterations after which we start taking samples
            thinning_period: number of iterations in one thinning period. we take
                            one sample from each period to decrease dependence 
                            between samples
            results_folder: folder to save results
            save_results: if True results are saved into a file named info_time.mcmc
            random_move: move used for proposing a new state is chosen randomly if True.
                            if False tries all moves on current state (until one is accepted)
                            in the order they are given in initial_state.moves
            move_probabilities: used if random_move is True. this a list of probabilities
                             for each move. Available moves are given in initial_state.moves
        """
        self.state = initial_state
        self.sampler_params = sampler_params
        self.info = sampler_params['info']
        self.run_count = sampler_params['runs']
        self.iter_count = sampler_params['iters']
        self.keep_top_n = sampler_params['keep_top_n']
        self.verbose = sampler_params['verbose']
        self.burn_in = sampler_params['burn_in']
        self.thinning_period = sampler_params['thinning_period']
        self.random_move = sampler_params['random_move']
        if self.random_move is True:
            self.move_probabilities = sampler_params['move_probabilities']
        self.best_samples = []
        self.best_probs = []
        self.samples = []
    
    def run(self):
        """
        Run sampler and return/save results
        """
        results = MCMCRunResults(self.info, self.state.__class__, self.sampler_params)
        for i in range(self.run_count):
            # reinitialize starting state for next run
            next_initial_state, acc_prob = self.state.moves[0]()
        
            results.initial_states[i] = deepcopy(self.state)
            samp, best_samp, acceptance_rate = self._run_once()
            results.samples[i] = deepcopy(samp)
            results.best_samples[i] = deepcopy(best_samp)
            results.acceptance_rate[i] = acceptance_rate
            
            self.state = next_initial_state
            
        results.end_time = time.localtime()
        if self.sampler_params['save_results'] is True:
            results.save()
        
        return results
        
    def _run_once(self):
        
        if self.verbose:
            print 'Starting sampling...'
        
        del self.best_probs[:]
        del self.best_samples[:]
        del self.samples[:]
        
        move_count = len(self.state.moves)
        current_move = 0
        last_move = 0
        accepted = 0
        for i in range(self.iter_count):
            
            if i > self.burn_in and i%self.thinning_period == 0:
                sample = MCMCSample(i, self.state, self.state.prior*self.state.likelihood, self.state.moves[last_move].__name__)
                self.samples.append(sample)
                if self.verbose:
                    print '\Sample: Adding {0:s}'.format(sample)
                            
            if self.random_move is True:
                chosen_move = np.random.choice(move_count, p=self.move_probabilities)
            else:
                chosen_move = current_move
            
            if self.verbose:
                if i%20 == 0:
                    print "Drawing sample {0:d} (move: {1:s})".format(i, self.state.moves[chosen_move].__name__)
            
            proposed_state, acceptance_prob = self.state.moves[chosen_move]()
            
            if np.random.rand() < acceptance_prob:
                # record the move that got us here
                last_move = chosen_move
                move_name = self.state.moves[last_move].__name__
                # if accepted, we should start from the first move in next iteration
                current_move = 0
                accepted = accepted + 1
                self.state = proposed_state
                if i > self.burn_in:
                    posterior = proposed_state.prior * proposed_state.likelihood
                    if  len(self.best_samples) < self.keep_top_n or posterior > np.min(self.best_probs):
                        # add this sample if we don't have it already in best samples
                        sample = MCMCSample(i, self.state, posterior, move_name)
                        if sample not in self.best_samples:
                            if self.verbose:
                                print '\tBest Sample: Adding {0:s}'.format(sample)
                            # find the index to which we need to insert this sample
                            ix = np.argwhere(posterior>self.best_probs)
                            if ix.size == 0:
                                ix = len(self.best_samples)
                            else:
                                ix = ix[0]
                            # insert sample    
                            self.best_probs.insert(ix, posterior)
                            self.best_samples.insert(ix, sample)
                            # remove last sample in list if we exceeded list size
                            if len(self.best_samples) > self.keep_top_n:
                                self.best_probs.pop()
                                self.best_samples.pop()
            else:
                # if not accepted, choose next move
                current_move = (current_move + 1) % move_count
        
        acceptance_rate = float(accepted) / self.iter_count
        return self.samples, self.best_samples, acceptance_rate

if __name__ == '__main__':
    # AoMR Shape Grammar, visual condition
    spatial_model = AoMRSpatialModel()
    forward_model = VisionForwardModel()
    data = np.load('data/visual/1.npy')
    state_params = {'b': 1250.0}
    init_state = AoMRShapeState(forward_model, data, state_params, spatial_model)
    sampler_params ={'info' : 'AoMR Shape Grammar - Visual Condition', 
                     'runs' : 2,
                     'iters' : 10000, 
                     'keep_top_n' : 20, 
                     'burn_in' : 0,
                     'thinning_period' : 400,
                     'results_folder' : './',
                     'save_results' : True,
                     'verbose': True}
     
    ms = MCMCSampler(sampler_params, init_state)
    results = ms.run()
    print(results)
    
    forward_model._view(results.best_samples[0][0].state)

#     # AoMR Shape Grammar, haptic condition
#     spatial_model = AoMRSpatialModel()
#     forward_model = HapticsForwardModel()
#     data = np.load('data/haptic/1.npy')
#     state_params = {'b': 5000.0}
#     init_state = AoMRShapeState(spatial_model, forward_model, data, state_params)
#     sampler_params = {'iters' : 2000, 
#                       'keep_top_n' : 20, 
#                       'burn_in' : 0,
#                       'thinning_period' : 100,
#                       'verbose': True}
#     ms = MCMCSampler(sampler_params, init_state)
#     samples, best_samples = ms.run()
#     
#     print 'Samples:'
#     for sample in samples:
#         print '\t' + repr(sample)
#         
#     print 'Best Samples:'
#     for sample in best_samples:
#         print '\t' + repr(sample)
#         
#     forward_model._view(best_samples[0].state)
#       

       
      
    