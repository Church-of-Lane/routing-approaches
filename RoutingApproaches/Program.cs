using System.Text.RegularExpressions;
using Python.Runtime;
using RoutingApproaches;
using Lane.Core.Messages;
using Lane.Core.Models;

const string laneUrl = "ws://204.168.211.4:5070";

// The name the model scores messages against, i.e. Lane's display name in the transcript.
string botName = Environment.GetEnvironmentVariable("ROUTING_BOT_NAME") ?? "Lane";

// Run from the project directory (as `dotnet run` does), so the Python folder sits beside us.
string pythonDir   = Path.GetFullPath(Environment.GetEnvironmentVariable("ROUTING_PYTHON_DIR") ?? "Python");
string weightsPath = Path.GetFullPath(Environment.GetEnvironmentVariable("ROUTING_WEIGHTS")   ?? Path.Combine(pythonDir, "routing.weights.h5"));
string tokensPath  = Path.GetFullPath(Environment.GetEnvironmentVariable("ROUTING_TOKEN_MAP") ?? Path.Combine(pythonDir, "token_map.json"));
string venvDir     = Path.Combine(pythonDir, ".venv");

ConfigurePythonRuntime(venvDir);
PythonEngine.Initialize();

PyObject model;
PyObject messageType;
int      history;
int      featureHistory;

using (Py.GIL())
{
    dynamic sys = Py.Import("sys");

    // The venv's packages (keras, numpy, ...) have to win over anything installed globally,
    // and machine_learning.py imports its siblings (util, symbolic) as top level modules.
    foreach (string sitePackages in Directory.GetDirectories(Path.Combine(venvDir, "lib"), "site-packages", SearchOption.AllDirectories))
        sys.path.insert(0, sitePackages);

    sys.path.insert(0, pythonDir);

    PyObject machineLearning = Py.Import("machine_learning");

    messageType    = Py.Import("util").GetAttr("Message");
    history        = machineLearning.GetAttr("HISTORY")        .As<int>();
    featureHistory = machineLearning.GetAttr("FEATURE_HISTORY").As<int>();

    Console.WriteLine($"Loading model from {weightsPath}");
    model = machineLearning.GetAttr("MachineLearning").Invoke(
        false.ToPython(), weightsPath.ToPython(), tokensPath.ToPython()
    );
}

// Initialize leaves the GIL held by this thread; release it so requests on other threads can take it.
PythonEngine.BeginAllowThreads();

Task<float> Inference(ModelRequest request, CancellationToken cancellationToken) => Task.Run(() =>
{
    List<(string? Author, string Text)> chat = [.. ParseTranscript(request.System), .. Incoming(request.Messages)];

    if (chat.Count == 0 || chat[^1].Author == botName)
        return 0f;

    // Only the last FEATURE_HISTORY messages are ever looked at.
    if (chat.Count > featureHistory)
        chat = chat[^featureHistory..];

    // The model needs at least HISTORY messages. Early in a conversation, pad the front with
    // blank, authorless messages, which read as "nothing happened here."
    while (chat.Count < history)
        chat.Insert(0, (null, ""));

    cancellationToken.ThrowIfCancellationRequested();

    using (Py.GIL())
    {
        using PyList messages = new();

        foreach ((string? author, string text) in chat)
        {
            using PyObject message = messageType.Invoke(
                author is null ? PyObject.None : author.ToPython(),
                new PyList(),
                text.ToPython(),
                false.ToPython()
            );

            messages.Append(message);
        }

        using PyObject score = model.InvokeMethod("inference", messages, botName.ToPython());

        return Math.Clamp(score.As<float>(), 0f, 1f);
    }
}, cancellationToken);

await Node.Connect(Inference, laneUrl);

// No PythonEngine.Shutdown() here: after the await we are on a thread pool thread rather than
// the one that initialized Python, and shutting down from there deadlocks. Exiting is enough.

// Lane puts the conversation so far into the system prompt, as lines of
// "[yyyy/MM/dd HH:mm:ss] Name: text" separated by blank lines, inside <transcript> tags.
static IEnumerable<(string? Author, string Text)> ParseTranscript(IEnumerable<PromptBlock> system)
{
    foreach (PromptBlock block in system)
    {
        Match transcript = Regex.Match(block.Text, @"<transcript>(.*?)</transcript>", RegexOptions.Singleline);
        if (!transcript.Success) continue;

        MatchCollection entries = Regex.Matches(
            transcript.Groups[1].Value,
            @"^\[\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}[^\]]*\] (?<author>.+?): (?<text>.*?)(?=^\[\d{4}/\d{2}/\d{2} |\z)",
            RegexOptions.Singleline | RegexOptions.Multiline
        );

        foreach (Match entry in entries)
        {
            string author = entry.Groups["author"].Value;

            // Lane's own inner monologue, not something said in the chat.
            if (author.EndsWith(" thinks")) continue;

            string text = entry.Groups["text"].Value.Trim().ReplaceLineEndings(" ");
            if (text.Length > 0)
                yield return (author, text);
        }
    }
}

// The messages that just arrived, and that are being routed (the last one especially.)
static IEnumerable<(string? Author, string Text)> Incoming(IEnumerable<LaneMessage> messages)
{
    foreach (LaneMessage message in messages)
    {
        string text = message.TextContent.Trim().ReplaceLineEndings(" ");
        if (text.Length > 0)
            yield return (message.Author.DisplayName, text);
    }
}

// Points pythonnet at the interpreter the venv was made from, unless PYTHONNET_PYDLL says otherwise.
static void ConfigurePythonRuntime(string venvDir)
{
    if (Environment.GetEnvironmentVariable("PYTHONNET_PYDLL") is { Length: > 0 } dll)
    {
        Runtime.PythonDLL = dll;
        return;
    }

    Dictionary<string, string> config = File.ReadAllLines(Path.Combine(venvDir, "pyvenv.cfg"))
        .Select (line => line.Split('=', 2))
        .Where  (parts => parts.Length == 2)
        .ToDictionary(parts => parts[0].Trim(), parts => parts[1].Trim());

    string[] version = config["version_info"].Split('.');
    string   home    = config["home"]; // the interpreter's bin directory

    Runtime.PythonDLL = Path.Combine(home, "..", "lib", $"libpython{version[0]}.{version[1]}.dylib");
}
