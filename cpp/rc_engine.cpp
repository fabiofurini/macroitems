// Strongest reduced costs over the dual optimal face, for all items of one region.
//
// Input: a digraph (V, E) of closure arcs (u, v) meaning "u in C implies v in C",
// and integer node values val with val(C) < 0 for every nonempty closed C (this is
// Proposition 3(a)/(c) of the paper: the null region Z with val = b, or the full
// region F with arcs reversed and val = -b).  Output, for every node i,
//     out[i] = min{ -val(C) : C closed, i in C }      (= s*_i or mu*_i, scaled).
//
// Picard's network: s -> v with capacity val_v > 0, v -> t with capacity -val_v,
// u -> v with infinite capacity for every closure arc.  Since the maximum closure is
// empty, the base maximum flow F0 saturates every source arc.  Forcing i into the
// closure adds an infinite arc s -> i, and out[i] = F_i - F0 is exactly the extra
// flow that can be pushed from i to t in the residual network of the base flow.
//
// Engines
//   0  "cold":     for every i, reset the flow and solve the forced problem from scratch (Dinic);
//   1  "warm":     base flow once; for every i, Dinic from i to t in the residual network,
//                  touching only the part reachable from i, then undo only the touched arcs;
//                  the search stops early at the upper bound -val(cl(i));
//   2  "goldberg": for every i, the Goldberg preflow code (third_party/flow_s.c), from scratch.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

using i64 = long long;

#ifdef WITH_GOLDBERG
extern "C" void goldberg(int, int, int, int, int*, int*, double*, double*, double*, int*, short*);
#endif

namespace {

// Residual network in CSR form: the arcs leaving u are first[u] .. first[u+1]-1, and
// rev[a] is the reverse arc of a.
struct Csr {
    int N = 0;
    std::vector<int> first, to, rev;
    std::vector<i64> cap;
    void build(int N_, const std::vector<int>& U, const std::vector<int>& V, const std::vector<i64>& C,
               std::vector<int>& idOf) {
        N = N_;
        const int M = (int)U.size();
        first.assign(N + 1, 0);
        for (int k = 0; k < M; ++k) { ++first[U[k] + 1]; ++first[V[k] + 1]; }
        for (int u = 0; u < N; ++u) first[u + 1] += first[u];
        std::vector<int> fill(first.begin(), first.end() - 1);
        to.assign(2 * M, 0); rev.assign(2 * M, 0); cap.assign(2 * M, 0); idOf.assign(M, 0);
        for (int k = 0; k < M; ++k) {
            int a = fill[U[k]]++, b = fill[V[k]]++;
            to[a] = V[k]; cap[a] = C[k]; rev[a] = b;
            to[b] = U[k]; cap[b] = 0;    rev[b] = a;
            idOf[k] = a;
        }
    }
};

// Dinic with time-stamped arrays (a call costs only the region it explores), an
// iterative blocking-flow search, a flow limit, and an undo log of touched arcs.
struct Dinic {
    Csr& g;
    std::vector<int> level, stamp, it, queue, path;
    std::vector<int> touched, savedStamp;
    std::vector<i64> saved;
    int curStamp = 0, saveEpoch = 0;
    bool logging = false;
    long long nPhases = 0, nVisited = 0;
    explicit Dinic(Csr& g_) : g(g_), level(g_.N), stamp(g_.N, -1), it(g_.N), queue(g_.N), path(g_.N + 1),
                              savedStamp(g_.cap.size(), -1) {}

    inline void touch(int a) {
        if (logging && savedStamp[a] != saveEpoch) {
            savedStamp[a] = saveEpoch; touched.push_back(a); saved.push_back(g.cap[a]);
        }
    }
    bool bfs(int s, int t, int blocked) {
        ++curStamp;
        int qh = 0, qt = 0;
        queue[qt++] = s; stamp[s] = curStamp; level[s] = 0;
        bool reached = false;
        while (qh < qt) {
            int u = queue[qh++];
            if (reached && level[u] >= level[t]) break;
            for (int a = g.first[u], e = g.first[u + 1]; a < e; ++a) {
                int v = g.to[a];
                if (g.cap[a] > 0 && v != blocked && stamp[v] != curStamp) {
                    stamp[v] = curStamp; level[v] = level[u] + 1;
                    if (v == t) reached = true; else queue[qt++] = v;
                }
            }
        }
        ++nPhases; nVisited += qt;
        if (reached) for (int k = 0; k < qt; ++k) it[queue[k]] = g.first[queue[k]];
        return reached;
    }
    // blocking flow, iterative; returns the flow pushed (at most limit)
    i64 blocking(int s, int t, int blocked, i64 limit) {
        i64 total = 0;
        int depth = 0, u = s;
        while (true) {
            if (u == t) {
                i64 f = limit - total;
                for (int k = 0; k < depth; ++k) f = std::min(f, g.cap[path[k]]);
                int cut = depth;
                for (int k = 0; k < depth; ++k) {
                    int a = path[k];
                    touch(a); touch(g.rev[a]);
                    g.cap[a] -= f; g.cap[g.rev[a]] += f;
                    if (g.cap[a] == 0 && cut == depth) cut = k;
                }
                total += f;
                if (total >= limit) return total;
                depth = cut;                                   // restart at the first saturated arc
                u = depth ? g.to[path[depth - 1]] : s;
                continue;
            }
            int& a = it[u];
            const int e = g.first[u + 1];
            while (a < e) {
                int v = g.to[a];
                if (g.cap[a] > 0 && v != blocked && stamp[v] == curStamp && level[v] == level[u] + 1) break;
                ++a;
            }
            if (a < e) { path[depth++] = a; u = g.to[a]; continue; }
            if (u == s) return total;                          // phase over
            level[u] = -1;                                     // dead end
            --depth;
            u = depth ? g.to[path[depth - 1]] : s;
            ++it[u];
        }
    }
    i64 maxflow(int s, int t, int blocked, i64 limit) {
        i64 total = 0;
        while (total < limit && bfs(s, t, blocked)) total += blocking(s, t, blocked, limit - total);
        return total;
    }
    void begin_log() { logging = true; ++saveEpoch; touched.clear(); saved.clear(); }
    void undo() {
        for (size_t k = 0; k < touched.size(); ++k) g.cap[touched[k]] = saved[k];
        logging = false;
    }
};


// Shortest augmenting path with exact distance labels computed once on the residual
// network of the base flow (Ahuja-Orlin), the gap rule, and an undo log for capacities,
// labels and label counts.  Every item restarts from the same base labels.
struct Sap {
    Csr& g;
    int N, s, t;
    std::vector<int> d, cnt, cur, curStamp, path;
    std::vector<int> touched, savedStamp, lblNode, lblOld;
    std::vector<i64> saved;
    int epoch = 0;
    long long relabels = 0, augments = 0;
    Sap(Csr& g_, int s_, int t_) : g(g_), N(g_.N), s(s_), t(t_), d(g_.N, g_.N), cnt(g_.N + 1, 0), cur(g_.N),
        curStamp(g_.N, -1), path(g_.N + 1), savedStamp(g_.cap.size(), -1) {}
    void exact_labels() {
        std::fill(d.begin(), d.end(), N);
        std::vector<int> q; q.reserve(N);
        d[t] = 0; q.push_back(t);
        for (size_t h = 0; h < q.size(); ++h) {
            int v = q[h];
            for (int b = g.first[v], e = g.first[v + 1]; b < e; ++b) {
                int u = g.to[b], a = g.rev[b];          // arc a: u -> v
                if (u != s && d[u] == N && g.cap[a] > 0) { d[u] = d[v] + 1; q.push_back(u); }
            }
        }
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int v = 0; v < N; ++v) if (v != s && d[v] < N) ++cnt[d[v]];
    }
    inline void touch(int a) {
        if (savedStamp[a] != epoch) { savedStamp[a] = epoch; touched.push_back(a); saved.push_back(g.cap[a]); }
    }
    // push from i to t up to limit; keep = true leaves the changes in place (logged)
    i64 run(int i, i64 limit, bool keep = false) {
        ++epoch;
        if (!keep) { touched.clear(); saved.clear(); lblNode.clear(); lblOld.clear(); }
        i64 total = 0;
        int depth = 0, u = i;
        while (d[i] < N) {
            if (u == t) {
                i64 f = limit - total;
                for (int k = 0; k < depth; ++k) f = std::min(f, g.cap[path[k]]);
                int cut = depth;
                for (int k = 0; k < depth; ++k) {
                    int a = path[k];
                    touch(a); touch(g.rev[a]);
                    g.cap[a] -= f; g.cap[g.rev[a]] += f;
                    if (g.cap[a] == 0 && cut == depth) cut = k;
                }
                ++augments; total += f;
                if (total >= limit) break;
                depth = cut; u = depth ? g.to[path[depth - 1]] : i;
                continue;
            }
            if (curStamp[u] != epoch) { curStamp[u] = epoch; cur[u] = g.first[u]; }
            int a = cur[u];
            const int e = g.first[u + 1];
            const int want = d[u] - 1;
            while (a < e && !(g.cap[a] > 0 && g.to[a] != s && d[g.to[a]] == want)) ++a;
            cur[u] = a;
            if (a < e) { path[depth++] = a; u = g.to[a]; continue; }
            // relabel u
            int mn = N;
            for (int b = g.first[u]; b < e; ++b)
                if (g.cap[b] > 0 && g.to[b] != s && d[g.to[b]] + 1 < mn) mn = d[g.to[b]] + 1;
            const int old = d[u];
            lblNode.push_back(u); lblOld.push_back(old);
            --cnt[old]; ++relabels;
            d[u] = mn; if (mn < N) ++cnt[mn];
            cur[u] = g.first[u];
            if (cnt[old] == 0) break;                  // gap: nothing above old reaches t
            if (u != i) { --depth; u = depth ? g.to[path[depth - 1]] : i; }
        }
        if (!keep) undo_to(0, 0);
        return total;
    }
    // undo, in reverse order, every change logged after the given marks
    void undo_to(size_t capMark, size_t lblMark) {
        for (size_t k = touched.size(); k-- > capMark;) g.cap[touched[k]] = saved[k];
        touched.resize(capMark); saved.resize(capMark);
        for (size_t k = lblNode.size(); k-- > lblMark;) {
            int v = lblNode[k];
            if (d[v] < N) --cnt[d[v]];
            d[v] = lblOld[k]; ++cnt[d[v]];
        }
        lblNode.resize(lblMark); lblOld.resize(lblMark);
    }
};

}  // namespace

extern "C" int rc_region(int n, int m, const int* tail, const int* head, const i64* val,
                         i64* out, int engine, double* stats) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int s = n, t = n + 1;
    i64 sumPos = 0, sumAbs = 0;
    for (int i = 0; i < n; ++i) { if (val[i] > 0) sumPos += val[i]; sumAbs += val[i] > 0 ? val[i] : -val[i]; }
    const i64 INF = sumAbs + 1;                    // exceeds every finite cut
    double tBase = 0, tItems = 0; i64 touchedTotal = 0; long long gPhases = 0, gVisited = 0;

    if (engine == 3) {                                 // warm start + exact labels (SAP)
        std::vector<int> U, V; std::vector<i64> C;
        for (int k = 0; k < m; ++k) { U.push_back(tail[k]); V.push_back(head[k]); C.push_back(INF); }
        for (int i = 0; i < n; ++i) {
            if (val[i] > 0) { U.push_back(s); V.push_back(i); C.push_back(val[i]); }
            else if (val[i] < 0) { U.push_back(i); V.push_back(t); C.push_back(-val[i]); }
        }
        Csr g; std::vector<int> idOf; g.build(n + 2, U, V, C, idOf);
        Dinic d(g);
        auto a = clk::now();
        i64 F0 = d.maxflow(s, t, -1, sumPos);
        if (F0 != sumPos) return -1;
        Sap sap(g, s, t); sap.exact_labels();
        tBase = std::chrono::duration<double>(clk::now() - a).count();
        std::vector<std::vector<int>> succ(n), pred(n);
        std::vector<int> indeg(n, 0);
        for (int k = 0; k < m; ++k) { succ[tail[k]].push_back(head[k]); pred[head[k]].push_back(tail[k]); ++indeg[head[k]]; }
        // Requirers before their prerequisites: if k requires i, every closed set containing
        // k contains i, so out[i] <= out[k].  Processing k first turns this into a bound
        // that is often exact and lets the search stop before the gap phase.
        std::vector<int> order; order.reserve(n);
        for (int i = 0; i < n; ++i) if (indeg[i] == 0) order.push_back(i);
        for (size_t h = 0; h < order.size(); ++h)
            for (int v : succ[order[h]]) if (--indeg[v] == 0) order.push_back(v);
        if ((int)order.size() != n) return -3;          // not acyclic
        std::vector<int> mark(n, -1), stack; stack.reserve(n);
        auto b = clk::now();
        long long tight = 0;
        for (int i : order) {
            i64 clv = 0; stack.clear(); stack.push_back(i); mark[i] = i;
            while (!stack.empty()) {
                int u = stack.back(); stack.pop_back(); clv += val[u];
                for (int v : succ[u]) if (mark[v] != i) { mark[v] = i; stack.push_back(v); }
            }
            i64 ub = -clv;
            for (int k : pred[i]) ub = std::min(ub, out[k]);
            out[i] = sap.run(i, ub);
            tight += (out[i] == ub);
        }
        tItems = std::chrono::duration<double>(clk::now() - b).count();
        gPhases = sap.augments; gVisited = sap.relabels; touchedTotal = tight;
    } else if (engine == 4) {                          // incremental along a prerequisite tree
        std::vector<int> U, V; std::vector<i64> C;
        for (int k = 0; k < m; ++k) { U.push_back(tail[k]); V.push_back(head[k]); C.push_back(INF); }
        for (int i = 0; i < n; ++i) {
            if (val[i] > 0) { U.push_back(s); V.push_back(i); C.push_back(val[i]); }
            else if (val[i] < 0) { U.push_back(i); V.push_back(t); C.push_back(-val[i]); }
        }
        Csr g; std::vector<int> idOf; g.build(n + 2, U, V, C, idOf);
        Dinic dn(g);
        auto a = clk::now();
        i64 F0 = dn.maxflow(s, t, -1, sumPos);
        if (F0 != sumPos) return -1;
        Sap sap(g, s, t); sap.exact_labels();
        tBase = std::chrono::duration<double>(clk::now() - a).count();
        // parent of k: one direct prerequisite (k requires parent); forcing k forces the parent,
        // so out[k] = out[parent] + (extra flow from k with the parent already forced)
        std::vector<std::vector<int>> succ(n), kids(n);
        std::vector<int> parent(n, -1), indeg(n, 0);
        for (int k = 0; k < m; ++k) { succ[tail[k]].push_back(head[k]); ++indeg[head[k]]; }
        // choose as parent the prerequisite with the largest closure (most work shared)
        std::vector<int> order; order.reserve(n);           // prerequisites before requirers
        { std::vector<int> outdeg(n, 0), q; std::vector<std::vector<int>> pr(n);
          for (int k = 0; k < m; ++k) { ++outdeg[tail[k]]; pr[head[k]].push_back(tail[k]); }
          for (int i = 0; i < n; ++i) if (outdeg[i] == 0) order.push_back(i);
          for (size_t h = 0; h < order.size(); ++h) for (int u : pr[order[h]]) if (--outdeg[u] == 0) order.push_back(u);
          if ((int)order.size() != n) return -3; }
        std::vector<int> depthOf(n, 0);
        for (int u : order) {
            int best = -1;
            for (int v : succ[u]) if (best < 0 || depthOf[v] > depthOf[best]) best = v;
            parent[u] = best;
            if (best >= 0) { depthOf[u] = depthOf[best] + 1; kids[best].push_back(u); }
        }
        // closure bound for every node (same as engine 3)
        std::vector<i64> clBound(n); std::vector<int> mark(n, -1), stk; stk.reserve(n);
        for (int i = 0; i < n; ++i) {
            i64 clv = 0; stk.clear(); stk.push_back(i); mark[i] = i;
            while (!stk.empty()) { int u = stk.back(); stk.pop_back(); clv += val[u];
                for (int v : succ[u]) if (mark[v] != i) { mark[v] = i; stk.push_back(v); } }
            clBound[i] = -clv;
        }
        auto b = clk::now();
        struct Frame { int node; size_t kid, capMark, lblMark; };
        std::vector<Frame> st;
        for (int r = 0; r < n; ++r) if (parent[r] < 0) {
            size_t cm = sap.touched.size(), lm = sap.lblNode.size();
            out[r] = sap.run(r, clBound[r], true);
            st.push_back({r, 0, cm, lm});
            while (!st.empty()) {
                Frame& f = st.back();
                if (f.kid < kids[f.node].size()) {
                    int k = kids[f.node][f.kid++];
                    size_t cm2 = sap.touched.size(), lm2 = sap.lblNode.size();
                    out[k] = out[f.node] + sap.run(k, clBound[k] - out[f.node], true);
                    st.push_back({k, 0, cm2, lm2});
                } else {
                    sap.undo_to(f.capMark, f.lblMark);
                    st.pop_back();
                }
            }
        }
        tItems = std::chrono::duration<double>(clk::now() - b).count();
        gPhases = sap.augments; gVisited = sap.relabels;
    } else if (engine == 0 || engine == 1) {
        std::vector<int> U, V; std::vector<i64> C;
        U.reserve(m + 2 * n); V.reserve(m + 2 * n); C.reserve(m + 2 * n);
        for (int k = 0; k < m; ++k) { U.push_back(tail[k]); V.push_back(head[k]); C.push_back(INF); }
        for (int i = 0; i < n; ++i) {
            if (val[i] > 0) { U.push_back(s); V.push_back(i); C.push_back(val[i]); }
            else if (val[i] < 0) { U.push_back(i); V.push_back(t); C.push_back(-val[i]); }
        }
        const int firstForce = (int)U.size();
        if (engine == 0) for (int i = 0; i < n; ++i) { U.push_back(s); V.push_back(i); C.push_back(0); }
        Csr g; std::vector<int> idOf;
        g.build(n + 2, U, V, C, idOf);
        Dinic d(g);
        auto a = clk::now();
        if (engine == 1) {
            i64 F0 = d.maxflow(s, t, -1, sumPos);
            tBase = std::chrono::duration<double>(clk::now() - a).count();
            if (F0 != sumPos) return -1;                       // hypothesis val(C) < 0 violated
            // upper bound -val(cl(i)) from the closure of i
            std::vector<std::vector<int>> succ(n);
            for (int k = 0; k < m; ++k) succ[tail[k]].push_back(head[k]);
            std::vector<int> mark(n, -1), stack; stack.reserve(n);
            auto b = clk::now();
            for (int i = 0; i < n; ++i) {
                i64 clv = 0; stack.clear(); stack.push_back(i); mark[i] = i;
                while (!stack.empty()) {
                    int u = stack.back(); stack.pop_back(); clv += val[u];
                    for (int v : succ[u]) if (mark[v] != i) { mark[v] = i; stack.push_back(v); }
                }
                d.begin_log();
                out[i] = d.maxflow(i, t, s, -clv);             // extra flow from i, never through s
                touchedTotal += (i64)d.touched.size();
                d.undo();
            }
            tItems = std::chrono::duration<double>(clk::now() - b).count();
            gPhases = d.nPhases; gVisited = d.nVisited;
        } else {
            std::vector<i64> cap0 = g.cap;
            i64 F0 = d.maxflow(s, t, -1, INF);
            if (F0 != sumPos) return -1;
            auto b = clk::now();
            for (int i = 0; i < n; ++i) {
                g.cap = cap0;
                int fa = idOf[firstForce + i];
                g.cap[fa] = INF;
                out[i] = d.maxflow(s, t, -1, INF) - F0;
            }
            tItems = std::chrono::duration<double>(clk::now() - b).count();
        }
    } else if (engine == 2) {
#ifdef WITH_GOLDBERG
        std::vector<int> fr, tt; std::vector<double> cp;
        for (int k = 0; k < m; ++k) { fr.push_back(tail[k] + 1); tt.push_back(head[k] + 1); cp.push_back((double)INF); }
        for (int i = 0; i < n; ++i) {
            if (val[i] > 0) { fr.push_back(n + 1); tt.push_back(i + 1); cp.push_back((double)val[i]); }
            else if (val[i] < 0) { fr.push_back(i + 1); tt.push_back(n + 2); cp.push_back((double)-val[i]); }
        }
        const int base = (int)fr.size();
        fr.push_back(n + 1); tt.push_back(1); cp.push_back((double)INF);
        double sumNeg = (double)(sumAbs - sumPos);
        auto b = clk::now();
        for (int i = 0; i < n; ++i) {
            tt[base] = i + 1;
            double mm = sumNeg + 1.0; int ncut = 0;
            goldberg(n + 2, base + 1, n + 1, n + 2, fr.data(), tt.data(), cp.data(), &mm, nullptr, &ncut, nullptr);
            out[i] = (i64)((mm - (double)sumPos) + (mm >= (double)sumPos ? 0.5 : -0.5));
        }
        tItems = std::chrono::duration<double>(clk::now() - b).count();
#else
        return -2;
#endif
    }
    if (stats) {
        stats[0] = tBase; stats[1] = tItems;
        stats[2] = std::chrono::duration<double>(clk::now() - t0).count();
        stats[3] = (double)touchedTotal;
        stats[4] = (double)gPhases; stats[5] = (double)gVisited;
    }
    return 0;
}
