import React, { useState } from 'react';
import { Shield, User, Lock, Monitor } from 'lucide-react';

interface LoginScreenProps {
  onLogin: (type: 'student' | 'admin') => void;
}

export default function LoginScreen({ onLogin }: LoginScreenProps) {
  const [selectedRole, setSelectedRole] = useState<'student' | 'admin'>('student');
  const [credentials, setCredentials] = useState({ username: '', password: '' });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (credentials.username && credentials.password) {
      onLogin(selectedRole);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 via-blue-800 to-teal-800">
      <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full mb-4">
  <img
    src="/assets/image.png"
    alt="Custom Icon"
    className="w-15 h-15"
  />
</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">EVA</h1>
          <p className="text-gray-600">Exam Vigilance Assistant</p>
        </div>

        <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
          <button
            type="button"
            onClick={() => setSelectedRole('student')}
            className={`flex-1 flex items-center justify-center py-2 px-4 rounded-md transition-all ${
              selectedRole === 'student'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:text-blue-600'
            }`}
          >
            <User className="w-4 h-4 mr-2" />
            Student
          </button>
          <button
            type="button"
            onClick={() => setSelectedRole('admin')}
            className={`flex-1 flex items-center justify-center py-2 px-4 rounded-md transition-all ${
              selectedRole === 'admin'
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-600 hover:text-blue-600'
            }`}
          >
            <Monitor className="w-4 h-4 mr-2" />
            Admin
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Username
            </label>
            <input
              type="text"
              value={credentials.username}
              onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder={selectedRole === 'student' ? 'Enter student ID' : 'Enter admin username'}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Password
            </label>
            <input
              type="password"
              value={credentials.password}
              onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              placeholder="Enter password"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center justify-center"
          >
            <Lock className="w-4 h-4 mr-2" />
            Secure Login
          </button>
        </form>

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h3 className="text-sm font-medium text-blue-900 mb-2">Security Features</h3>
          <ul className="text-xs text-blue-700 space-y-1">
            <li>• Real-time AI monitoring</li>
            <li>• Webcam & audio surveillance</li>
            <li>• Tab switching detection</li>
            <li>• Copy-paste prevention</li>
          </ul>
        </div>
      </div>
    </div>
  );
}