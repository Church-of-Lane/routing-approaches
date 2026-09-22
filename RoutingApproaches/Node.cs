using System.Diagnostics;
using System.Text.Json;
using Lane.Core.Messages;
using Lane.Core.Models;
using Lane.Node.Sdk;

namespace RoutingApproaches
{
    /// <summary>
    /// Sets up a node to connect the approach selected to Lane.
    /// </summary>
    public static class Node
    {
        /// <summary>
        /// Describes a method for responding to a given request, with a float describing the
        /// routing score (between 0 and 1.)
        /// </summary>
        public delegate Task<float> Responder(ModelRequest request, CancellationToken ct);

        // For this project, we don't care about the emoticon generated. Typically this is still
        // done by the router, but for our purposes we will just keep it at some default.
        private const string EMOTICON = ":)";

        /// <summary>
        /// Connect the node to some Lane instance (running at <paramref name="laneUrl"/>), and use
        /// <paramref name="responder"/> to respond to requests. Only listens for routing requests.
        /// </summary>
        public static async Task Connect(Responder responder, string laneUrl)
        {
            // so that ctrl+C stops the program
            using CancellationTokenSource stop = new();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };

            // creates an identifying key for the node
            using FileNodeKey key = FileNodeKey.LoadOrCreate("node-key.pem");
            LaneNode node = new(
                new LaneNodeOptions
                {
                    LaneUrl = laneUrl,
                    Model   = "routing-approaches",
                    Pool    = "cheap", // Cheap refers to routing,
                    Name    = "Routing approaches"
                },
                key,
                async (request, ct) =>
                {
                    Stopwatch stopwatch = new();
                    stopwatch.Start();

                    float response = await responder(request, ct);

                    stopwatch.Stop();

                    return new ModelResponse(
                        [
                            new ToolUsePart(
                                Guid.NewGuid().ToString("n"),
                                "assess",
                                JsonSerializer.SerializeToElement(new {
                                    enthusiasm = response, 
                                    emoticon   = EMOTICON 
                                })
                            )
                        ],
                        StopReason.ToolUse,
                        // once we set up the LLM approach we might add in actual token
                        // usage stats here, but it's not an important metric (I think) for
                        // this project.
                        new TokenUsage(0, 0, 0, 0, "node", stopwatch.Elapsed)
                    );
                }
            );

            node.Connected    += connection => Console.WriteLine($"Connected as {node.Identity.KeyId} ({connection})");
            node.Disconnected += reason     => Console.WriteLine($"Disconnected: {reason}");

            await node.RunAsync(stop.Token);
        }
    }
}