from __future__ import annotations

import copy
import itertools
import math
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import nashpy as nash
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD

from pycid.core.cpd import DecisionDomain, FunctionCPD
from pycid.core.macid_base import AgentLabel, MACIDBase
from pycid.core.relevance_graph import CondensedRelevanceGraph


class MACID(MACIDBase):
    """A Multi-Agent Causal Influence Diagram"""

    def get_all_pure_ne(self) -> List[List[FunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in the MACID.
        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_all_pure_ne_in_sg()

    def joint_pure_policies(self, decisions: Iterable[str]) -> List[Tuple[FunctionCPD, ...]]:
        all_dec_decision_rules = list(map(self.pure_decision_rules, decisions))
        return list(itertools.product(*all_dec_decision_rules))

    def get_all_pure_ne_in_sg(self, decisions_in_sg: Optional[Iterable[str]] = None) -> List[List[FunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in a MACID subgame.

        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MAID subgame.
        - If decisions_in_sg is not specified, this method finds all pure NE in the full MACID.
        - If the MACID being operated on already has function CPDs for some decision nodes, it is
        assumed that these have already been optimised and so these are not changed.
        """
        # TODO: Check that the decisions in decisions_in_sg actually make up a MAID subgame
        if decisions_in_sg is None:
            decisions_in_sg = self.decisions
        else:
            decisions_in_sg = set(decisions_in_sg)  # For efficient membership checks

        agents_in_sg = list({self.decision_agent[dec] for dec in decisions_in_sg})
        agent_decs_in_sg = {
            agent: [dec for dec in self.agent_decisions[agent] if dec in decisions_in_sg] for agent in agents_in_sg
        }

        # impute random decisions to non-instantiated, irrelevant decision nodes
        macid = self.copy()
        for d in macid.decisions:
            if not macid.is_s_reachable(decisions_in_sg, d) and isinstance(macid.get_cpds(d), DecisionDomain):
                macid.impute_random_decision(d)

        # NE finder
        all_pure_ne_in_sg: List[List[FunctionCPD]] = []
        for pp in self.joint_pure_policies(decisions_in_sg):
            macid.add_cpds(*pp)  # impute the policy profile

            for a in agents_in_sg:  # check that each agent is happy
                eu_pp_agent_a = macid.expected_utility({}, agent=a)
                macid.add_cpds(*macid.optimal_pure_policies(agent_decs_in_sg[a])[0])
                max_eu_agent_a = macid.expected_utility({}, agent=a)

                if max_eu_agent_a > eu_pp_agent_a:  # not an NE
                    break
            else:  # it's an NE
                all_pure_ne_in_sg.append(list(pp))

        return all_pure_ne_in_sg

    def policy_profile_assignment(self, partial_policy: Iterable[FunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        pp: Dict[str, Optional[TabularCPD]] = {d: None for d in self.decisions}
        pp.update({cpd.variable: cpd for cpd in partial_policy})
        return pp

    def get_all_pure_spe(self) -> List[List[FunctionCPD]]:
        """Return a list of all pure subgame perfect Nash equilbiria (SPE) in the MACIM
        - Each SPE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        spes: List[List[FunctionCPD]] = [[]]

        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(CondensedRelevanceGraph(self).get_scc_topological_ordering()):
            extended_spes = []
            for partial_profile in spes:
                self.add_cpds(*partial_profile)
                all_ne_in_sg = self.get_all_pure_ne_in_sg(scc)
                for ne in all_ne_in_sg:
                    extended_spes.append(partial_profile + list(ne))
            spes = extended_spes
        return spes

    def get_mixed_ne(self) -> List[List[TabularCPD]]:
        """
        Returns all mixed Nash equilibria in non-degnerate 2-player games using
        Nashpy's support enumeration implementation.

        Returns most mixed Nash equilbria in degnerate 2-player games.
        """
        agents = list(self.agents)
        if len(agents) != 2:
            raise ValueError(
                f"This MACID has {len(agents)} agents and yet this method currently only works for 2 agent games."
            )

        agent_pure_policies = [list(self.pure_policies(self.agent_decisions[agent])) for agent in agents]

        def _agent_util(pp: Tuple[FunctionCPD, ...], agent: AgentLabel) -> float:
            self.add_cpds(*pp)
            return self.expected_utility({}, agent=agent)

        payoff1 = np.array(
            [[_agent_util(pp1 + pp2, agents[0]) for pp2 in agent_pure_policies[1]] for pp1 in agent_pure_policies[0]]
        )
        payoff2 = np.array(
            [[_agent_util(pp1 + pp2, agents[1]) for pp2 in agent_pure_policies[1]] for pp1 in agent_pure_policies[0]]
        )
        game = nash.Game(payoff1, payoff2)  # could make cleaner with yield instead
        equilibria = game.support_enumeration()

        all_mixed_ne = []
        for eq in equilibria:
            mixed_ne = list(
                itertools.chain(
                    *[list(self.mixed_policy_cpd(agent_pure_policies[agent], eq[agent])) for agent in range(2)]
                )
            )
            all_mixed_ne.append(mixed_ne)
        return all_mixed_ne

    def mixed_policy_cpd(
        self, pure_policies: List[Tuple[FunctionCPD, ...]], prob_dist: List[float]
    ) -> Iterator[TabularCPD]:
        """
        #TODO it should be returning FunctionCPDs rather than TabularCPDs...
        Return a mixed policy cpd.

        Parameters
        ----------
        pure_policies = a List of all pure policies for a certain agent in a MACID.
        prob_dist = the mixed policy's probability distribution over pure policies.

        """
        if not math.isclose(sum(prob_dist), 1.0, abs_tol=0.01):
            raise ValueError(f"The values in {prob_dist} do not sum to 1")

        num_decision_rules = len(pure_policies[0])  # how many decision nodes that agent has

        for i in range(num_decision_rules):
            variable = pure_policies[0][i].variable
            card = len(pure_policies[0][i].domain)  # type: ignore
            evidence = self.get_parents(variable)
            evidence_card = [self.get_cardinality(variable) for variable in evidence]

            pure_prop_matrices = [pure_policy[i].values for pure_policy in pure_policies]
            new_mixed_prob_matrix = np.array(
                sum([matrix * prob for matrix, prob in zip(pure_prop_matrices, prob_dist)])
            )

            if new_mixed_prob_matrix.ndim == 1:  # annoying pgmpy feature
                new_mixed_prob_matrix = np.atleast_2d(new_mixed_prob_matrix).reshape(-1, 1)

            decision_rule = TabularCPD(
                variable,
                card,
                new_mixed_prob_matrix,
                evidence,
                evidence_card,
                state_names={variable: pure_policies[0][i].domain},
            )
            yield decision_rule

    def decs_in_each_maid_subgame(self) -> List[set]:
        """
        Return a list giving the set of decision nodes in each MAID subgame of the original MAID.
        """
        con_rel = CondensedRelevanceGraph(self)
        con_rel_sccs = con_rel.nodes  # the nodes of the condensed relevance graph are the maximal sccs of the MA(C)ID
        powerset = list(
            itertools.chain.from_iterable(
                itertools.combinations(con_rel_sccs, r) for r in range(1, len(con_rel_sccs) + 1)
            )
        )
        con_rel_subgames = copy.deepcopy(powerset)
        for subset in powerset:
            for node in subset:
                if not nx.descendants(con_rel, node).issubset(subset) and subset in con_rel_subgames:
                    con_rel_subgames.remove(subset)

        dec_subgames = [
            [con_rel.get_decisions_in_scc()[scc] for scc in con_rel_subgame] for con_rel_subgame in con_rel_subgames
        ]

        return [set(itertools.chain.from_iterable(i)) for i in dec_subgames]

    def copy_without_cpds(self) -> MACID:
        """copy the MACID structure"""
        return MACID(
            edges=self.edges(),
            agent_decisions=self.agent_decisions,
            agent_utilities=self.agent_utilities,
        )
