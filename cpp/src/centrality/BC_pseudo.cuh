// algorithm Brandes(Graph) is
//     for each u in Graph.Vertices do 
//         CB[u] <-- 0

//     for each s in Graph.Vertices do
//         for each v in Graph.Vertices do
//             delta[v] <-- 0 // single dependency of s on v --> delta is dependencies (how much source s depends on vertex v for shortest paths)
//             prev[v] <-- empty list // immediate predecessors of v during BFS
//             sigma[v] <-- 0 // number of shortest paths from s to v (s implied) --> sigma is count
//             dist[v] <-- null // no paths known initially
//         sigma[s] <-- 1 // 1 count for start vertex
//         dist[s] <-- 0

//         Q <-- queue containing only s // BFS
//         S <-- empty stack // Record the order in which vertices are visited --> used for backtracking later

//         // SSSP

//         while Q is not empty do
//             u <-- Q.dequeue()
//             S.push(u) // add to stack --> stack S records order vertices are discovered 

//             for each v in Graph.Neighbours[u] do
//                 if dist[v] = null then 
//                     dist[v] <-- dist[u] + 1
//                     Q.enqueue(v)
//                 if dist[v] = dist[u] + 1 then // found another shortest path. dist[v] < dist[u] + 1 should never happen in BFS. dist[v] > dist[u] + 1 means non-shortest path found so skip it.
//                     sigma[v] <-- sigma[v] + sigma[u] // path counting --> tracks # of shortest paths from s to v. When we first discover vertex v, sigma[v] = 0, then we add sigma[u] to it. If we discover v again from another predecessor u' at the same distance, we add sigma[u'] to it. Basically, if vertex v is reachable from any vertex u, then we add # of shortest paths to reach vertex u to total # of paths that can go from source to vertex v.
//                     prev[v].append(u) // predecessor tracking --> tracks immediate predecessors of v for backpropagation

//         // Backpropagation of dependencies

//         while S is not empty do
//             v <-- S.pop() // process vertices in reverse order of discovery (LIFO w/ stack)

//             for each u in prev[v] do 
//                 delta[u] <-- delta[u] + sigma[u] / sigma[v] * (1 + delta[v])

//             if v != s then // skip source vertex because it doesn't contribute to its own betweenness centrality
//                 CB[v] <-- CB[v] + delta[v] // add dependency score delta[v] to vertex v's BC score

//     return CB

#include <queue>
#include <stack>
#include <vector>
using namespace std;
            

vector<int> Brandes(Graph& graph) {
    vector<int> CB(graph.numVertices);
    for(int u = 0; u < graph.numVertices; u++){
        CB[u] = 0;
    }

    int delta[graph.numVertices];
    vector<int> prev[graph.numVertices];
    int sigma[graph.numVertices];
    int dist[graph.numVertices];

    for(int s = 0; s < graph.numVertices; s++){
        for(int v = 0; v < graph.numVertices; v++){ // need to set this for all v's connected to this particular source s
            delta[v] = 0;
            prev[v] = vector<int>();
            sigma[v] = 0;
            dist[v] = -1; // -1 means no path known yet
        }

        // this is attribute for sources themselves so no need to put in loop of vertices v tht are connected to sources
        sigma[s] = 1;
        dist[s] = 0;

        queue<int> Q;
        Q.push(s);

        stack<int> S;

        while(!Q.empty()){
            int u = Q.front();
            Q.pop();
            S.push(u); 

            for(int v : graph.Neighbours[u]){
                if(dist[v] == -1){
                    dist[v] = dist[u] + 1;
                    Q.push(v);
                }
                if(dist[v] == dist[u] + 1){
                    sigma[v] += sigma[u];
                    prev[v].push_back(u);
                }
            }
        }

        while(!S.empty()){
            int v = S.top();
            S.pop();
            
            for(int u : prev[v]){
                delta[u] += sigma[u] / sigma[v] * (1 + delta[v]);
            }
            if(v != s){
                CB[v] += delta[v];
            }
        }
    }
    return CB;
}
        