
import axios from 'axios';
 
import React,{Component} from 'react';
 
class App extends Component {
  
    state = {
 
      // Initially, no file is selected
      selectedFile: null
    };
    
    // On file select (from the pop up)
    onFileChange = event => {
    
      // Update the state
      this.setState({ selectedFile: event.target.files[0] });
    
    };
    
    // On file upload (click the upload button)
    onFileUpload = () => {
    
      // Create an object of formData
      const formData = new FormData();
    
      // Update the formData object
      formData.append(
        "myFile",
        this.state.selectedFile,
        this.state.selectedFile.name
      );
    
      // Details of the uploaded file
      console.log(this.state.selectedFile);
      axios.post("http://localhost:5000/test", formData)
      .then(response => {
        console.log(response)
        // response --> image
      } )
      .catch(error => {
        console.log(error)
      })
    };
    
    fileData = () => {
    
      if (this.state.selectedFile) {
         
        return (
          <div>
            {this.state.selectedFile && <ImageThumb image={this.state.selectedFile} />}
          </div>
        );
      } else {
        return (
          <div>
            <br />
            <h4>Choose before Pressing the Upload button</h4>
          </div>
        );
      }
    };
    
    render() {
    
      return (

        
        <div>
            <div className="sidebar__title" style={{padding: '20px', paddingTop: '130px'}}>
                  <div className="sidebar__image">
                      <h3 style={{fontSize: '26px'}}>Select a Satellite Image</h3>
                  </div>
            
          {this.fileData()}
          </div>
                <input type="file" onChange={this.onFileChange} style={{paddingLeft: '20px', alignContent: "center", justifyContent: "center", display:"flex"}} />
                <div className="App" style={{paddingLeft: '20px', alignContent: "center", justifyContent: "center", display:"flex"}}>
                      <button type="button" id = "assess" onClick={this.onFileUpload} style={{backgroundColor:"#fff000", padding: '10px', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold', padding: '10px 15px', textAlign: 'center', transition: '100ms', maxWidth:'180px', boxSizing:'border-box', border: '0', fontSize:'16px', userSelect:'none', WebkitUserSelect:'none', touchAction:'manipulation',marginTop:'15px'}}> ASSESS </button>
               </div> 
          
        </div>
        
      );
    }
    
  }
 
  const ImageThumb = ({ image }) => {
    return <img src={URL.createObjectURL(image)} alt={image.name} style={{width:'480px', height:'320px', padding: '20px'}} />;
};
  export default App;


