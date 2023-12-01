#include once
#include <chrono>
#include <iostream>
#include <stack>
#include <unordered_map>

#include <mujincontrollercommon/logging.h>
#include <mujincontrollercommon/mujincontrollercommon.h>

// Define a structure to represent a function call
struct FunctionCall {
    std::string functionName;
    std::chrono::steady_clock::time_point startTime;
};

// Define a structure for each node in the radix tree
struct RadixTreeNode {
    std::string functionName;
    std::chrono::microseconds totalTime;
    int callCount;
    std::unordered_map<std::string, RadixTreeNode> children;
};

class Profiler {
public:
    /// A singleton.
    static StopWatch& me() {
        static StopWatch instance;
        return instance;
    }

    Profiler() {
        // Initialize the call stack with a dummy root node
        callStack.push({"Root", std::chrono::steady_clock::now()});
    }

    // Called when a function starts
    void FunctionStart(const std::string& functionName) {
        auto startTime = std::chrono::steady_clock::now();
        callStack.push({functionName, startTime});
    }

    // Called when a function ends
    void FunctionEnd() {
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - callStack.top().startTime);

        // Update the radix tree with profiling information
        UpdateRadixTree(elapsedTime);

        // Pop the function from the call stack
        callStack.pop();
    }

    // Print the profiling information
    void PrintProfilingData(const RadixTreeNode& node, int depth = 0) {
        for (int i = 0; i < depth; ++i) {
            std::cout << "  ";
        }

        std::cout << node.functionName << " - Total Time: " << node.totalTime.count() << " us, Call Count: " << node.callCount << std::endl;

        for (const auto& child : node.children) {
            PrintProfilingData(child.second, depth + 1);
        }
    }

private:
    std::stack<FunctionCall> callStack;
    RadixTreeNode radixTreeRoot;

    // Update the radix tree with profiling information
    void UpdateRadixTree(const std::chrono::microseconds& elapsedTime) {
        auto currentNode = &radixTreeRoot;

        while (!callStack.empty()) {
            auto& functionName = callStack.top().functionName;

            // Create a new node if the function is not in the radix tree
            if (currentNode->children.find(functionName) == currentNode->children.end()) {
                currentNode->children[functionName] = {functionName, std::chrono::microseconds::zero(), 0};
            }

            // Update profiling information
            currentNode = &currentNode->children[functionName];
            currentNode->totalTime += elapsedTime;
            currentNode->callCount++;

            // Move to the parent node
            callStack.pop();

            // Break if the call stack is empty
            if (callStack.empty()) {
                break;
            }
        }
    }
};

/// Utility to time how long a scope takes
class ScopeTimer {
public:
    ScopeTimer(Profiler& profiler, ::log4cxx::LoggerPtr& logger_, const char* scopeName, const char* fileName, int lineNumber)
        : logger(logger_), _scopeName(scopeName), _fileName(fileName), _lineNumber(lineNumber) {
        _startTimestampUS = mujincontrollercommon::GetMonotonicTime();
    }

    ~ScopeTimer() {
        auto now = std::chrono::high_resolution_clock::now();

        // get caller thread name
        char threadName[16] = {};
        pthread_getname_np(pthread_self(), threadName, sizeof(threadName));
        threadName[sizeof(threadName) - 1] = '\0';

        MUJIN_LOG_DEBUG_FORMAT("%s:%d: Scope '%s' took %d[us], calling thread is '%s'", _fileName % _lineNumber % _scopeName % elapsedTimeUS % threadName);
    }

    inline uint64_t GetStartTimestampUS() const {
        return _startTimestampUS;
    }

private:
    /// Pointer to scope-local logger, since we can't assume access to one in the header
    ::log4cxx::LoggerPtr& logger;

    /// User-defined scope name
    const char* _scopeName = nullptr;

    /// Call site of this timer (file)
    const char* _fileName = nullptr;

    /// Call site of this timer (line)
    int _lineNumber = 0;

    /// Recorded scope entry timestamp
    uint64_t _startTimestampUS = 0;
};

#define TIME_THIS_FUNCTION() ScopeTimer functionTimer(logger, __FUNCTION__, __FILE__, __LINE__)
#define TIME_THIS_SCOPE(scopeName) ScopeTimer __scopeTimer##__COUNTER__(logger, scopeName, __FILE__, __LINE__)

int main() {
    Profiler profiler;

    FunctionA(profiler);
    FunctionB(profiler);

    // Print profiling data
    profiler.PrintProfilingData(profiler.GetRadixTreeRoot());

    return 0;
}
