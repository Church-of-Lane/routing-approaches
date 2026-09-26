using Python.Runtime;
using RoutingApproaches;

PythonEngine.Initialize();

const string laneUrl = "ws://204.168.211.4:5070";

await Node.Connect((x, y) => Task.FromResult(1f), laneUrl);
