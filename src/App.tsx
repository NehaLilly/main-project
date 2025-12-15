import React, { useState, useEffect } from 'react';
import { Shield, Monitor, Code, Users, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import ExamInterface from './components/ExamInterface';
import AdminDashboard from './components/AdminDashboard';
import LoginScreen from './components/LoginScreen';
import { ExamProvider } from './context/ExamContext';

function App() {
  const [currentView, setCurrentView] = useState<'login' | 'exam' | 'admin'>('login');
  const [userType, setUserType] = useState<'student' | 'admin' | null>(null);

  const handleLogin = (type: 'student' | 'admin') => {
    setUserType(type);
    setCurrentView(type === 'student' ? 'exam' : 'admin');
  };

  const handleLogout = () => {
    setCurrentView('login');
    setUserType(null);
  };

  return (
    <ExamProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        {currentView === 'login' && (
          <LoginScreen onLogin={handleLogin} />
        )}
        
        {currentView === 'exam' && (
          <ExamInterface onLogout={handleLogout} />
        )}
        
        {currentView === 'admin' && (
          <AdminDashboard onLogout={handleLogout} />
        )}
      </div>
    </ExamProvider>
  );
}

export default App;