import React, { useState } from 'react';
import { Play, Save, RotateCcw, Terminal, CheckCircle, XCircle } from 'lucide-react';
import { useExam } from '../context/ExamContext';

export default function CodeEditor() {
  const { submitCode } = useExam();
  const [language, setLanguage] = useState('javascript');
  const [code, setCode] = useState(`// Implement your binary search function here
function binarySearch(arr, target) {
    // Your code here
    
}

// Test your function
console.log(binarySearch([1, 3, 5, 7, 9, 11], 7)); // Expected: 3
console.log(binarySearch([1, 3, 5, 7, 9, 11], 4)); // Expected: -1`);
  
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const handleRunCode = async () => {
    setIsRunning(true);
    
    // Simulate code execution
    setTimeout(() => {
      try {
        // Simple simulation of JavaScript execution
        if (language === 'javascript') {
          const logs: string[] = [];
          const mockConsole = {
            log: (...args: any[]) => logs.push(args.join(' '))
          };
          
          // Execute code in a safe context (this is a simplified simulation)
          const result = new Function('console', code).call(null, mockConsole);
          setOutput(logs.join('\n') || 'Code executed successfully');
        } else {
          setOutput('Code execution simulated for ' + language);
        }
      } catch (error) {
        setOutput(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
      setIsRunning(false);
    }, 1500);
  };

  const handleSubmitCode = () => {
    submitCode(code, language);
    setOutput('Code submitted successfully!');
  };

  const handleResetCode = () => {
    setCode(`// Implement your binary search function here
function binarySearch(arr, target) {
    // Your code here
    
}

// Test your function
console.log(binarySearch([1, 3, 5, 7, 9, 11], 7)); // Expected: 3
console.log(binarySearch([1, 3, 5, 7, 9, 11], 4)); // Expected: -1`);
    setOutput('');
  };

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600 text-sm focus:ring-2 focus:ring-blue-500"
            >
              <option value="javascript">JavaScript</option>
              <option value="python">Python</option>
              <option value="java">Java</option>
              <option value="cpp">C++</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleRunCode}
              disabled={isRunning}
              className="flex items-center space-x-2 bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              <span>{isRunning ? 'Running...' : 'Run'}</span>
            </button>

            <button
              onClick={handleSubmitCode}
              className="flex items-center space-x-2 bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>Submit</span>
            </button>

            <button
              onClick={handleResetCode}
              className="flex items-center space-x-2 bg-gray-600 text-white px-3 py-1 rounded text-sm hover:bg-gray-700 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>
        </div>
      </div>

      {/* Code Editor */}
      <div className="flex-1 flex flex-col">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="code-editor flex-1 bg-gray-900 text-gray-100 p-4 font-mono text-sm leading-relaxed focus:outline-none resize-none"
          style={{ 
            fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
            tabSize: 2
          }}
          placeholder="Write your code here..."
          spellCheck={false}
        />

        {/* Output Panel */}
        <div className="h-48 bg-gray-800 border-t border-gray-700">
          <div className="px-4 py-2 border-b border-gray-700 flex items-center space-x-2">
            <Terminal className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Output</span>
            {output && (
              <div className="flex items-center space-x-1">
                {output.includes('Error') ? (
                  <XCircle className="w-4 h-4 text-red-500" />
                ) : (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                )}
              </div>
            )}
          </div>
          <div className="p-4 h-40 overflow-auto">
            <pre className="text-gray-300 text-sm font-mono whitespace-pre-wrap">
              {output || 'Click "Run" to execute your code...'}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}