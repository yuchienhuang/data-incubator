function newListItem(name,content) {
  const item = document.createElement('li');
  item.className = name;  
  item.appendChild(content);
  return item
}

function newNavbarItem(text, url) {
  const itemLink = document.createElement('a');
  itemLink.className = 'nav-item nav-link';
  itemLink.innerHTML = text;
  itemLink.href = url;

  return itemLink
}



function renderNavbarList(user) {
  const navbarList = document.getElementById("mm-nav");

  // navbarList.appendChild(newListItem("",newNavbarItem('Home', '/')));
  navbarList.appendChild(newListItem("",newNavbarItem('Plot 1', '/plot1')));
  navbarList.appendChild(newListItem("",newNavbarItem('Plot 2', '/plot2')));

  
  // navbarList.appendChild(newListItem("",newNavbarItem('Profile','/u/profile?'+user._id)));


  }

