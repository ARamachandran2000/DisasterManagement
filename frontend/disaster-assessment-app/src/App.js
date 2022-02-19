import logo from './logo.svg';
import './App.css';
import {useState} from 'react';
import Navbar from "./components/navbar/Navbar";
import Sidebar from "./components/sidebar/Sidebar";
import Main from "./components/main/Main";

const Dashboard= () => {
  const [sidebarOpen,setSidebarOpen]=useState(false);
  const openSidebar = () => {
    setSidebarOpen(true);
  };
  const closeSidebar = () =>{
    setSidebarOpen(false);
  };
  return (
    <div className="container">
     
      <Main/>
      <Sidebar sidebarOpen={sidebarOpen} closeSidebar={closeSidebar}/>
    </div>
  );
}

export default Dashboard;

