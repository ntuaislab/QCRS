import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from ipdb import set_trace as st
import scipy.stats as stats
import random
from dataclasses import dataclass, field


'''
to delete
the certify function return pabar additionally.

'''
@dataclass()
class Smooth(object):
    base_classifier: torch.nn.Module 
    num_classes: int
    
    def __post_init__(self):
        self.ABSTAIN = -1

    # def __init__(self, base_classifier: torch.nn.Module, num_classes: int):
    #     """
    #     :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
    #     :param num_classes:
    #     """
    #     self.base_classifier = base_classifier
    #     self.num_classes = num_classes
    
    def binary_search_step(self, x, n, epsilon, init_lambda, original_sigma):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, n, original_sigma)
        cAHat = counts_selection.argmax().item()
        nA = counts_selection[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, 0.001)
        if pABar < 0.5:
            r12 = 0
        else:
            r12 = original_sigma * norm.ppf(pABar)
            
        step = epsilon/2
        current_sigma = original_sigma
        direction = 0
        while init_lambda > epsilon:
            gradient, radius_value = self.take_gradient(x, current_sigma, step, n, cAHat)
            if np.sign(gradient) > 0:
                current_sigma = current_sigma + init_lambda
                direction = 1
            elif np.sign(gradient) < 0:
                current_sigma = current_sigma - init_lambda
                direction = -1
            else:
                if np.abs(direction-0) < 10e-5:
                    current_sigma = current_sigma + init_lambda
                    direction = 1
                elif direction > 0:
                    current_sigma = current_sigma - init_lambda
                    direction = -1
                else:
                    current_sigma = current_sigma + init_lambda
                    direction = 1
            init_lambda = init_lambda*0.75
        
        _, original_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=original_sigma)
        _, proposed_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=current_sigma)
        if original_radius>= proposed_radius:
            return original_sigma
        else:
            return current_sigma
    
    def binary_search_bisection(self, x, n, epsilon, left_sigma, right_sigma, original_sigma, epsilon_step):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, n, original_sigma)
        cAHat = counts_selection.argmax().item()
        nA = counts_selection[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, 0.001)
        if pABar < 0.5:
            r12 = 0
        else:
            r12 = original_sigma * norm.ppf(pABar)
            
        step = epsilon/epsilon_step
        direction = 0
        
        current_sigma = (left_sigma + right_sigma) / 2
        gradient, radius_value = self.take_gradient(x, current_sigma, step, n, cAHat)
        if np.sign(gradient) > 0:
            left_sigma = current_sigma
            direction = 1
        elif np.sign(gradient) < 0:
            right_sigma = current_sigma
            direction = -1
        else:
            right_sigma = current_sigma
            direction = -1

        while right_sigma-left_sigma > epsilon:
            current_sigma = (right_sigma + left_sigma) / 2
            gradient, radius_value = self.take_gradient(x, current_sigma, step, n, cAHat)
            if np.sign(gradient) > 0:
                left_sigma = current_sigma
                direction = 1
            elif np.sign(gradient) < 0:
                right_sigma = current_sigma
                direction = -1
            else:
                if np.abs(direction-0) < 10e-5:
                    right_sigma = current_sigma
                    direction = -1
                elif direction > 0:
                    right_sigma = current_sigma
                    direction = -1
                else:
                    left_sigma = current_sigma
                    direction = 1
                    
        current_sigma = (right_sigma + left_sigma) / 2
        _, original_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=original_sigma)
        _, proposed_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=current_sigma)
        if original_radius >= proposed_radius:
            return original_sigma
        else:
            return current_sigma

    
    def binary_search(self, x, n, epsilon, left, right, original_sigma):
        """
        To search the optimal sigma that provides the best radius
        parameters:
            x: the input [channel x height x width]
            n: number of samples used to estimate the gradient
            epsilon: to achieve epsilon-suboptimal
            left and right: the sigma searching region 
        """
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, n, original_sigma)
        cAHat = counts_selection.argmax().item()
        nA = counts_selection[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, 0.001)
        if pABar < 0.5:
            r12 = 0
        else:
            r12 = original_sigma * norm.ppf(pABar)

        step = epsilon/2
        # while np.abs(right-left) >= epsilon:
        # st()
        left_gradient, left_value = self.take_gradient(x, left, step, n, cAHat)
        right_gradient, right_value = self.take_gradient(x, right, step, n, cAHat)
        failure_side = 0
        if left_gradient == 0 and right_gradient == 0:
            return original_sigma
        if left_gradient == 0:
            left_gradient = +1
            failure_side = -1
        if right_gradient == 0:
            right_gradient = -1
            failure_side = +1
        if np.sign(left_gradient) == np.sign(right_gradient):
            if left_value >= right_value: # or to check the sign is negative
                # if np.sign(left_gradient)>0: #
                    # st() #
                return left
            else:
                return right
        # st()
        cnt = 0
        while np.abs(right-left) > epsilon:
            # st()
            cnt += 1
            if cnt > 20: break
            center = np.round((left+right)/2, 4)
            center_gradient, _ = self.take_gradient(x, center, step, n, cAHat)
            if center_gradient<0:
                left = center
            elif center_gradient>0:
                right = center
            else:
                if failure_side<0:
                    left = center
                if failure_side>0:
                    right = center
        proposed_sigma = np.round((left+right)/2, 4)
        _, original_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=original_sigma)
        _, proposed_radius, _ = self.fast_certify(x, n=n, alpha=0.001, batch_size=n, sigma=proposed_sigma)
        if original_radius>= proposed_radius:
            return original_sigma
        else:
            return proposed_sigma

    def take_gradient(self, x, sigma, step, sample_size, cAHat):
        left_cAhat, left_radius, left_pABar = self.fast_certify(x, n=sample_size, alpha=0.001, batch_size=sample_size, sigma=sigma-step)
        right_cAhat, right_radius, right_pABar = self.fast_certify(x, n=sample_size, alpha=0.001, batch_size=sample_size, sigma=sigma+step)
        if left_cAhat != cAHat:
            left_radius = 0
        if right_cAhat != cAHat:
            right_radius = 0
        gradient = right_radius - left_radius
        return gradient, left_radius


    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, sigma: float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        # st()
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, sigma)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, sigma)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return self.ABSTAIN, 0.0, pABar
        else:
            radius = sigma * norm.ppf(pABar)
            return cAHat, radius, pABar

    def fast_certify(self, x: torch.tensor, n: int, alpha: float, batch_size: int, sigma: float):
        """ Revised function certify, without counts_estimation.  
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, batch_size, sigma)
        cAHat = counts_selection.argmax().item()
        nA = counts_selection[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return self.ABSTAIN, 0.0, pABar
        else:
            radius = sigma * norm.ppf(pABar)
            return cAHat, radius, pABar


    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int, sigma):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size, sigma)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return self.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, sigma):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size 

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * sigma
                # st()
                # st here they did not clip the image 
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float):
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
