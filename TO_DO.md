13 June 2013
*----DONE------
*Grammar classes should support different moves (proposals).
*MCMC sampler should be able to use these different moves with different strategies
	*Possible moves can be given as a list to MCMCSampler at init or better: Grammar classes
		can have a moves attribute which is a list of proposal functions
	*We can also specify how to use these rules; such as use one rule randomly in each iteration, 
		or use all of them consecutively on the current state in each iteration
*---DONE-------
*Acceptance probabilities should depend on proposal, now there's a single acceptance prob. method,
	but depending on the move used acceptance probabilities may be different.
	
14 June 2013
*----DONE------
*Implement new moves for AoMR Shape Grammar. Possible moves: Add/remove branch, change part, 
	change part location
	
20 June 2013
*-----CANCELED FOR NOW------------------
*I need to do the acceptance probability calculation in MCMCSampler using prior, likelihood and
proposal probabilities. Because if I want to do annealing I need the nominator and denominator in
acceptance probability separately. I should probably return proposal probabilities for both the 
proposed move and its reverse from proposal function, then use already available prior and ll values
with these proposal probabilities to calculate acceptance probabilities inside sampler.

*----DONE------
5 July 2013
*Implement tree kernels (subtree kernel for now) for calculating similarity between samples
*Implement P-kernel
**Add these kernels as methods to shape_grammar instance.
