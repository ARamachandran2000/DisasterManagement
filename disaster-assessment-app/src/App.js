import logo from './logo.svg';
import './App.css';
import {useState} from 'react';
import Navbar from "./components/navbar/Navbar";
import Sidebar from "./components/sidebar/Sidebar";
import Main from "./components/main/Main";

const Dashboard= () => {
  const [sidebarOpen,setSidebarOpen]=useState(false);
  const [image,setImage]=useState(null);
  const [areaDamage, setAreaDamage] = useState(null);
  const openSidebar = () => {
    setSidebarOpen(true);
  };
  const closeSidebar = () =>{
    setSidebarOpen(false);
  };
  return (
    <div className="container">
     
      <Main image={image} areaDamage={areaDamage}/>
      <Sidebar sidebarOpen={sidebarOpen} closeSidebar={closeSidebar} name="H" changeImage={setImage} changeDamageVal = {setAreaDamage}/>
    </div>
  );
}

export default Dashboard;

