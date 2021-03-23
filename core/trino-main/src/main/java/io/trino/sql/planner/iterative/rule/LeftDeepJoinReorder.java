/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.sql.planner.iterative.rule;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import io.airlift.log.Logger;
import io.trino.Session;
import io.trino.cost.CostComparator;
import io.trino.cost.CostProvider;
import io.trino.cost.PlanCostEstimate;
import io.trino.cost.PlanNodeStatsAndCostSummary;
import io.trino.cost.PlanNodeStatsEstimate;
import io.trino.cost.StatsProvider;
import io.trino.matching.Capture;
import io.trino.matching.Captures;
import io.trino.matching.Pattern;
import io.trino.metadata.Metadata;
import io.trino.operator.LookupJoinOperator;
import io.trino.sql.analyzer.FeaturesConfig.JoinDistributionType;
import io.trino.sql.planner.EqualityInference;
import io.trino.sql.planner.PlanNodeIdAllocator;
import io.trino.sql.planner.Symbol;
import io.trino.sql.planner.SymbolsExtractor;
import io.trino.sql.planner.iterative.Lookup;
import io.trino.sql.planner.iterative.Rule;
import io.trino.sql.planner.plan.*;
import io.trino.sql.planner.plan.JoinNode.DistributionType;
import io.trino.sql.planner.plan.JoinNode.EquiJoinClause;
import io.trino.sql.tree.ComparisonExpression;
import io.trino.sql.tree.Expression;
import io.trino.sql.tree.SymbolReference;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.IntStream;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Verify.verify;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.Sets.powerSet;
import static io.trino.SystemSessionProperties.getJoinDistributionType;
import static io.trino.SystemSessionProperties.getJoinReorderingStrategy;
import static io.trino.SystemSessionProperties.getMaxReorderedJoins;
import static io.trino.matching.Capture.newCapture;
import static io.trino.sql.ExpressionUtils.and;
import static io.trino.sql.ExpressionUtils.combineConjuncts;
import static io.trino.sql.ExpressionUtils.extractConjuncts;
import static io.trino.sql.analyzer.FeaturesConfig.JoinReorderingStrategy.AUTOMATIC;
import static io.trino.sql.planner.DeterminismEvaluator.isDeterministic;
import static io.trino.sql.planner.EqualityInference.nonInferrableConjuncts;
import static io.trino.sql.planner.iterative.Plans.resolveGroupReferences;
import static io.trino.sql.planner.iterative.rule.DetermineJoinDistributionType.canReplicate;
import static io.trino.sql.planner.iterative.rule.PushProjectionThroughJoin.pushProjectionThroughJoin;
import static io.trino.sql.planner.iterative.rule.LeftDeepJoinReorder.JoinEnumerationResult.INFINITE_COST_RESULT;
import static io.trino.sql.planner.iterative.rule.LeftDeepJoinReorder.JoinEnumerationResult.UNKNOWN_COST_RESULT;
import static io.trino.sql.planner.iterative.rule.LeftDeepJoinReorder.MultiJoinNode.toMultiJoinNode;
import static io.trino.sql.planner.optimizations.QueryCardinalityUtil.isAtMostScalar;
import static io.trino.sql.planner.plan.JoinNode.DistributionType.PARTITIONED;
import static io.trino.sql.planner.plan.JoinNode.DistributionType.REPLICATED;
import static io.trino.sql.planner.plan.JoinNode.Type.INNER;
import static io.trino.sql.planner.plan.JoinNode.Type.LEFT;
import static io.trino.sql.planner.plan.Patterns.*;
import static io.trino.sql.planner.plan.Patterns.join;
import static io.trino.sql.tree.BooleanLiteral.TRUE_LITERAL;
import static io.trino.sql.tree.ComparisonExpression.Operator.EQUAL;
import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.toCollection;


public class LeftDeepJoinReorder
        implements Rule<JoinNode>
{
    private static final Logger log = Logger.get(LeftDeepJoinReorder.class);

    private static final Capture<JoinNode> CHILD = newCapture();
    // We check that join distribution type is absent because we only want
    // to do this transformation once (reordered joins will have distribution type already set).
    private final Pattern<JoinNode> pattern;

    private final Metadata metadata;
    private final CostComparator costComparator;

    public LeftDeepJoinReorder(Metadata metadata, CostComparator costComparator)
    {
        this.metadata = requireNonNull(metadata, "metadata is null");
        this.costComparator = requireNonNull(costComparator, "costComparator is null");
        this.pattern = join()
                .matching((planNode, lookup) ->
                        leftDeepJoinPattern(planNode, ((Lookup) lookup), 1));
    }

    public boolean leftDeepJoinPattern(PlanNode planNode, Lookup lookup, int depth) {
        // TODO: mark explored subtrees
        if (planNode instanceof JoinNode) {
            JoinNode joinNode = (JoinNode) planNode;
            if (joinNode.getDistributionType().isEmpty()
                    && isDeterministic(joinNode.getFilter().orElse(TRUE_LITERAL), metadata)) {
                PlanNode left = lookup.resolve(joinNode.getLeft());
                if (left instanceof JoinNode) {
                    return leftDeepJoinPattern(left, lookup, depth + 1);
                }
            }
        }
        return depth != 1;
    }

    @Override
    public Pattern<JoinNode> getPattern()
    {
        return pattern;
    }

    @Override
    public boolean isEnabled(Session session)
    {
        return getJoinReorderingStrategy(session) == AUTOMATIC;
    }

    @Override
    public Result apply(JoinNode joinNode, Captures captures, Context context)
    {
        // try to reorder
        LeftDeepJoinReorder.MultiJoinNode multiJoinNode = toMultiJoinNode(metadata, joinNode, context);
        // choose join order
        LeftDeepJoinReorder.JoinEnumerator joinEnumerator = new LeftDeepJoinReorder.JoinEnumerator(
                metadata,
                costComparator,
                multiJoinNode.getFilter(),
                context);
        LeftDeepJoinReorder.JoinEnumerationResult result =  joinEnumerator.chooseJoinOrder(
                multiJoinNode.getSources(),
                multiJoinNode.getOutputSymbols());

        // TODO: check using isEmpty().
        return Result.ofPlanNode(result.getPlanNode().get());
    }

    @VisibleForTesting
    static class JoinEnumerator
    {
        private final Metadata metadata;
        private final Session session;
        private final StatsProvider statsProvider;
        private final CostProvider costProvider;
        // Using Ordering to facilitate rule determinism
        private final Ordering<LeftDeepJoinReorder.JoinEnumerationResult> resultComparator;
        private final PlanNodeIdAllocator idAllocator;
        private final Expression allFilter;
        private final EqualityInference allFilterInference;
        private final Lookup lookup;
        private final Context context;
        // A memo for dynamic-programming
        private final Map<Set<PlanNode>, LeftDeepJoinReorder.JoinEnumerationResult> memo = new HashMap<>();

        @VisibleForTesting
        JoinEnumerator(Metadata metadata, CostComparator costComparator, Expression filter, Context context)
        {
            this.metadata = requireNonNull(metadata, "metadata is null");
            this.context = requireNonNull(context);
            this.session = requireNonNull(context.getSession(), "session is null");
            this.statsProvider = requireNonNull(context.getStatsProvider(), "statsProvider is null");
            this.costProvider = requireNonNull(context.getCostProvider(), "costProvider is null");
            this.resultComparator = costComparator.forSession(session).onResultOf(result -> result.cost);
            this.idAllocator = requireNonNull(context.getIdAllocator(), "idAllocator is null");
            this.allFilter = requireNonNull(filter, "filter is null");
            this.allFilterInference = EqualityInference.newInstance(metadata, filter);
            this.lookup = requireNonNull(context.getLookup(), "lookup is null");
        }

        private LeftDeepJoinReorder.JoinEnumerationResult chooseJoinOrder(LinkedHashSet<PlanNode> sources, List<Symbol> outputSymbols)
        {
            // TODO: This method choose the best order of the `sources`(input tables for the join tree)
            // TODO: might need timeout checking
            //context.checkTimeoutNotExhausted();
            // construct partition = {1} for testing
            Set<Integer> partition = IntStream.range(1, 2)
                    .boxed()
                    .collect(toImmutableSet());
            checkState(sources.size() > 1, "sources size is less than or equal to one");
            // TODO: Now using `partition` that partition the sources into left ({1}) and right (sources - {1})
            // TODO: and `createJoinAccordingToPartitioning` calls `createJoin` calls `getJoinSource`,
            // TODO: which will recursively calls `chooseJoinOrder`, and finally give a complete join plan
            LeftDeepJoinReorder.JoinEnumerationResult result = createJoinAccordingToPartitioning(sources, outputSymbols, partition);

            result.planNode.ifPresent((planNode) -> log.debug("Least cost join was: %s", planNode));
            return result;
        }

        @VisibleForTesting
        LeftDeepJoinReorder.JoinEnumerationResult createJoinAccordingToPartitioning(LinkedHashSet<PlanNode> sources, List<Symbol> outputSymbols, Set<Integer> partitioning)
        {
            List<PlanNode> sourceList = ImmutableList.copyOf(sources);
            LinkedHashSet<PlanNode> leftSources = partitioning.stream()
                    .map(sourceList::get)
                    .collect(toCollection(LinkedHashSet::new));
            LinkedHashSet<PlanNode> rightSources = sources.stream()
                    .filter(source -> !leftSources.contains(source))
                    .collect(toCollection(LinkedHashSet::new));
            return createJoin(leftSources, rightSources, outputSymbols);
        }

        private LeftDeepJoinReorder.JoinEnumerationResult createJoin(LinkedHashSet<PlanNode> leftSources, LinkedHashSet<PlanNode> rightSources, List<Symbol> outputSymbols)
        {
            Set<Symbol> leftSymbols = leftSources.stream()
                    .flatMap(node -> node.getOutputSymbols().stream())
                    .collect(toImmutableSet());
            Set<Symbol> rightSymbols = rightSources.stream()
                    .flatMap(node -> node.getOutputSymbols().stream())
                    .collect(toImmutableSet());

            List<Expression> joinPredicates = getJoinPredicates(leftSymbols, rightSymbols);
            List<EquiJoinClause> joinConditions = joinPredicates.stream()
                    .filter(LeftDeepJoinReorder.JoinEnumerator::isJoinEqualityCondition)
                    .map(predicate -> toEquiJoinClause((ComparisonExpression) predicate, leftSymbols))
                    .collect(toImmutableList());

            List<Expression> joinFilters = joinPredicates.stream()
                    .filter(predicate -> !isJoinEqualityCondition(predicate))
                    .collect(toImmutableList());

            Set<Symbol> requiredJoinSymbols = ImmutableSet.<Symbol>builder()
                    .addAll(outputSymbols)
                    .addAll(SymbolsExtractor.extractUnique(joinPredicates))
                    .build();

            LeftDeepJoinReorder.JoinEnumerationResult leftResult = getJoinSource(
                    leftSources,
                    requiredJoinSymbols.stream()
                            .filter(leftSymbols::contains)
                            .collect(toImmutableList()));

            PlanNode left = leftResult.planNode.orElseThrow(() -> new VerifyException("Plan node is not present"));

            LeftDeepJoinReorder.JoinEnumerationResult rightResult = getJoinSource(
                    rightSources,
                    requiredJoinSymbols.stream()
                            .filter(rightSymbols::contains)
                            .collect(toImmutableList()));

            PlanNode right = rightResult.planNode.orElseThrow(() -> new VerifyException("Plan node is not present"));

            List<Symbol> leftOutputSymbols = left.getOutputSymbols().stream()
                    .filter(outputSymbols::contains)
                    .collect(toImmutableList());
            List<Symbol> rightOutputSymbols = right.getOutputSymbols().stream()
                    .filter(outputSymbols::contains)
                    .collect(toImmutableList());

            return setJoinNodeProperties(new JoinNode(
                    idAllocator.getNextId(),
                    INNER,
                    left,
                    right,
                    joinConditions,
                    leftOutputSymbols,
                    rightOutputSymbols,
                    false,
                    joinFilters.isEmpty() ? Optional.empty() : Optional.of(and(joinFilters)),
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty(),
                    ImmutableMap.of(),
                    Optional.empty()));
        }

        private List<Expression> getJoinPredicates(Set<Symbol> leftSymbols, Set<Symbol> rightSymbols)
        {
            ImmutableList.Builder<Expression> joinPredicatesBuilder = ImmutableList.builder();

            // This takes all conjuncts that were part of allFilters that
            // could not be used for equality inference.
            // If they use both the left and right symbols, we add them to the list of joinPredicates
            nonInferrableConjuncts(metadata, allFilter).stream()
                    .map(conjunct -> allFilterInference.rewrite(conjunct, Sets.union(leftSymbols, rightSymbols)))
                    .filter(Objects::nonNull)
                    // filter expressions that contain only left or right symbols
                    .filter(conjunct -> allFilterInference.rewrite(conjunct, leftSymbols) == null)
                    .filter(conjunct -> allFilterInference.rewrite(conjunct, rightSymbols) == null)
                    .forEach(joinPredicatesBuilder::add);

            // create equality inference on available symbols
            // TODO: make generateEqualitiesPartitionedBy take left and right scope
            List<Expression> joinEqualities = allFilterInference.generateEqualitiesPartitionedBy(Sets.union(leftSymbols, rightSymbols)).getScopeEqualities();
            EqualityInference joinInference = EqualityInference.newInstance(metadata, joinEqualities.toArray(new Expression[0]));
            joinPredicatesBuilder.addAll(joinInference.generateEqualitiesPartitionedBy(leftSymbols).getScopeStraddlingEqualities());

            return joinPredicatesBuilder.build();
        }

        private LeftDeepJoinReorder.JoinEnumerationResult getJoinSource(LinkedHashSet<PlanNode> nodes, List<Symbol> outputSymbols)
        {
            if (nodes.size() == 1) {
                PlanNode planNode = getOnlyElement(nodes);
                Set<Symbol> scope = ImmutableSet.copyOf(outputSymbols);
                ImmutableList.Builder<Expression> predicates = ImmutableList.builder();
                predicates.addAll(allFilterInference.generateEqualitiesPartitionedBy(scope).getScopeEqualities());
                nonInferrableConjuncts(metadata, allFilter).stream()
                        .map(conjunct -> allFilterInference.rewrite(conjunct, scope))
                        .filter(Objects::nonNull)
                        .forEach(predicates::add);
                Expression filter = combineConjuncts(metadata, predicates.build());
                if (!TRUE_LITERAL.equals(filter)) {
                    planNode = new FilterNode(idAllocator.getNextId(), planNode, filter);
                }
                return createJoinEnumerationResult(planNode);
            }
            return chooseJoinOrder(nodes, outputSymbols);
        }

        private static boolean isJoinEqualityCondition(Expression expression)
        {
            return expression instanceof ComparisonExpression
                    && ((ComparisonExpression) expression).getOperator() == EQUAL
                    && ((ComparisonExpression) expression).getLeft() instanceof SymbolReference
                    && ((ComparisonExpression) expression).getRight() instanceof SymbolReference;
        }

        private static EquiJoinClause toEquiJoinClause(ComparisonExpression equality, Set<Symbol> leftSymbols)
        {
            Symbol leftSymbol = Symbol.from(equality.getLeft());
            Symbol rightSymbol = Symbol.from(equality.getRight());
            EquiJoinClause equiJoinClause = new EquiJoinClause(leftSymbol, rightSymbol);
            return leftSymbols.contains(leftSymbol) ? equiJoinClause : equiJoinClause.flip();
        }

        private LeftDeepJoinReorder.JoinEnumerationResult setJoinNodeProperties(JoinNode joinNode)
        {
            if (isAtMostScalar(joinNode.getRight(), lookup)) {
                return createJoinEnumerationResult(joinNode.withDistributionType(REPLICATED));
            }
            if (isAtMostScalar(joinNode.getLeft(), lookup)) {
                return createJoinEnumerationResult(joinNode.flipChildren().withDistributionType(REPLICATED));
            }
            // TODO: testing
//            return new JoinEnumerationResult(
//                    Optional.of(joinNode),
//                    costProvider.getCost(lookup.resolve(joinNode.getLeft())));
            return createJoinEnumerationResult(joinNode);
        }

        private LeftDeepJoinReorder.JoinEnumerationResult createJoinEnumerationResult(JoinNode joinNode)
        {
            PlanCostEstimate costEstimate = costProvider.getCost(joinNode);
            PlanNodeStatsEstimate statsEstimate = statsProvider.getStats(joinNode);
            return LeftDeepJoinReorder.JoinEnumerationResult.createJoinEnumerationResult(
                    Optional.of(joinNode.withReorderJoinStatsAndCost(new PlanNodeStatsAndCostSummary(
                            statsEstimate.getOutputRowCount(),
                            statsEstimate.getOutputSizeInBytes(joinNode.getOutputSymbols(), context.getSymbolAllocator().getTypes()),
                            costEstimate.getCpuCost(),
                            costEstimate.getMaxMemory(),
                            costEstimate.getNetworkCost()))),
                    costEstimate);
        }

        private LeftDeepJoinReorder.JoinEnumerationResult createJoinEnumerationResult(PlanNode planNode)
        {
            return LeftDeepJoinReorder.JoinEnumerationResult.createJoinEnumerationResult(Optional.of(planNode), costProvider.getCost(planNode));
        }
    }

    /**
     * This class represents a set of inner joins that can be executed in any order.
     */
    @VisibleForTesting
    static class MultiJoinNode
    {
        // Use a linked hash set to ensure optimizer is deterministic
        private final LinkedHashSet<PlanNode> sources;
        private final Expression filter;
        private final List<Symbol> outputSymbols;

        MultiJoinNode(LinkedHashSet<PlanNode> sources, Expression filter, List<Symbol> outputSymbols)
        {
            requireNonNull(sources, "sources is null");
            checkArgument(sources.size() > 1, "sources size is <= 1");
            requireNonNull(filter, "filter is null");
            requireNonNull(outputSymbols, "outputSymbols is null");

            this.sources = sources;
            this.filter = filter;
            this.outputSymbols = ImmutableList.copyOf(outputSymbols);

            List<Symbol> inputSymbols = sources.stream().flatMap(source -> source.getOutputSymbols().stream()).collect(toImmutableList());
            checkArgument(inputSymbols.containsAll(outputSymbols), "inputs do not contain all output symbols");
        }

        public Expression getFilter()
        {
            return filter;
        }

        public LinkedHashSet<PlanNode> getSources()
        {
            return sources;
        }

        public List<Symbol> getOutputSymbols()
        {
            return outputSymbols;
        }

        public static LeftDeepJoinReorder.MultiJoinNode.Builder builder()
        {
            return new LeftDeepJoinReorder.MultiJoinNode.Builder();
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(sources, ImmutableSet.copyOf(extractConjuncts(filter)), outputSymbols);
        }

        @Override
        public boolean equals(Object obj)
        {
            if (!(obj instanceof LeftDeepJoinReorder.MultiJoinNode)) {
                return false;
            }

            LeftDeepJoinReorder.MultiJoinNode other = (LeftDeepJoinReorder.MultiJoinNode) obj;
            return this.sources.equals(other.sources)
                    && ImmutableSet.copyOf(extractConjuncts(this.filter)).equals(ImmutableSet.copyOf(extractConjuncts(other.filter)))
                    && this.outputSymbols.equals(other.outputSymbols);
        }

        static LeftDeepJoinReorder.MultiJoinNode toMultiJoinNode(Metadata metadata, JoinNode joinNode, Context context)
        {
            return toMultiJoinNode(metadata, joinNode, context.getLookup(), context.getIdAllocator(), getMaxReorderedJoins(context.getSession()));
        }

        static LeftDeepJoinReorder.MultiJoinNode toMultiJoinNode(Metadata metadata, JoinNode joinNode, Lookup lookup, PlanNodeIdAllocator planNodeIdAllocator, int joinLimit)
        {
            // the number of sources is the number of joins + 1
            return new LeftDeepJoinReorder.MultiJoinNode.JoinNodeFlattener(metadata, joinNode, lookup, planNodeIdAllocator, joinLimit + 1).toMultiJoinNode();
        }

        private static class JoinNodeFlattener
        {
            private final Metadata metadata;
            private final Lookup lookup;
            private final PlanNodeIdAllocator planNodeIdAllocator;

            private final LinkedHashSet<PlanNode> sources = new LinkedHashSet<>();
            private final List<Expression> filters = new ArrayList<>();
            private final List<Symbol> outputSymbols;

            JoinNodeFlattener(Metadata metadata, JoinNode node, Lookup lookup, PlanNodeIdAllocator planNodeIdAllocator, int sourceLimit)
            {
                this.metadata = requireNonNull(metadata, "metadata is null");
                requireNonNull(node, "node is null");
                this.outputSymbols = node.getOutputSymbols();
                this.lookup = requireNonNull(lookup, "lookup is null");
                this.planNodeIdAllocator = requireNonNull(planNodeIdAllocator, "planNodeIdAllocator is null");
                flattenNode(node, sourceLimit);
            }

            private void flattenNode(PlanNode node, int limit)
            {
                PlanNode resolved = lookup.resolve(node);

                // (limit - 2) because you need to account for adding left and right side
                if (!(resolved instanceof JoinNode) || (sources.size() > (limit - 2))) {
                    sources.add(resolved);
                    return;
                }

                JoinNode joinNode = (JoinNode) resolved;

                // we set the left limit to limit - 1 to account for the node on the right
                flattenNode(joinNode.getLeft(), limit - 1);
                flattenNode(joinNode.getRight(), limit);
                joinNode.getCriteria().stream()
                        .map(EquiJoinClause::toExpression)
                        .forEach(filters::add);
                joinNode.getFilter().ifPresent(filters::add);
            }

            LeftDeepJoinReorder.MultiJoinNode toMultiJoinNode()
            {
                return new LeftDeepJoinReorder.MultiJoinNode(sources, and(filters), outputSymbols);
            }
        }

        static class Builder
        {
            private List<PlanNode> sources;
            private Expression filter;
            private List<Symbol> outputSymbols;

            public LeftDeepJoinReorder.MultiJoinNode.Builder setSources(PlanNode... sources)
            {
                this.sources = ImmutableList.copyOf(sources);
                return this;
            }

            public LeftDeepJoinReorder.MultiJoinNode.Builder setFilter(Expression filter)
            {
                this.filter = filter;
                return this;
            }

            public LeftDeepJoinReorder.MultiJoinNode.Builder setOutputSymbols(Symbol... outputSymbols)
            {
                this.outputSymbols = ImmutableList.copyOf(outputSymbols);
                return this;
            }

            public LeftDeepJoinReorder.MultiJoinNode build()
            {
                return new LeftDeepJoinReorder.MultiJoinNode(new LinkedHashSet<>(sources), filter, outputSymbols);
            }
        }
    }

    @VisibleForTesting
    static class JoinEnumerationResult
    {
        static final LeftDeepJoinReorder.JoinEnumerationResult UNKNOWN_COST_RESULT = new LeftDeepJoinReorder.JoinEnumerationResult(Optional.empty(), PlanCostEstimate.unknown());
        static final LeftDeepJoinReorder.JoinEnumerationResult INFINITE_COST_RESULT = new LeftDeepJoinReorder.JoinEnumerationResult(Optional.empty(), PlanCostEstimate.infinite());

        private final Optional<PlanNode> planNode;
        private final PlanCostEstimate cost;

        private JoinEnumerationResult(Optional<PlanNode> planNode, PlanCostEstimate cost)
        {
            this.planNode = requireNonNull(planNode, "planNode is null");
            this.cost = requireNonNull(cost, "cost is null");
            checkArgument((cost.hasUnknownComponents() || cost.equals(PlanCostEstimate.infinite())) && planNode.isEmpty()
                            || (!cost.hasUnknownComponents() || !cost.equals(PlanCostEstimate.infinite())) && planNode.isPresent(),
                    "planNode should be present if and only if cost is known");
        }

        public Optional<PlanNode> getPlanNode()
        {
            return planNode;
        }

        public PlanCostEstimate getCost()
        {
            return cost;
        }

        static LeftDeepJoinReorder.JoinEnumerationResult createJoinEnumerationResult(Optional<PlanNode> planNode, PlanCostEstimate cost)
        {
            if (cost.hasUnknownComponents()) {
                return UNKNOWN_COST_RESULT;
            }
            if (cost.equals(PlanCostEstimate.infinite())) {
                return INFINITE_COST_RESULT;
            }
            return new LeftDeepJoinReorder.JoinEnumerationResult(planNode, cost);
        }
    }
}

