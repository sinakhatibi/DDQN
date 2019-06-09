<!DOCTYPE html>
<html class="" lang="en">
<head prefix="og: http://ogp.me/ns#">
<meta charset="utf-8">
<link href="https://assets.gitlab-static.net" rel="dns-prefetch">
<link crossorigin="" href="https://assets.gitlab-static.net" rel="preconnnect">
<meta content="IE=edge" http-equiv="X-UA-Compatible">
<meta content="object" property="og:type">
<meta content="GitLab" property="og:site_name">
<meta content="AI/DDQN.py · 5GM-WP4 · Sina Khatibi / SDNController" property="og:title">
<meta content="GitLab.com" property="og:description">
<meta content="https://assets.gitlab-static.net/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="og:image">
<meta content="64" property="og:image:width">
<meta content="64" property="og:image:height">
<meta content="https://gitlab.com/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py" property="og:url">
<meta content="summary" property="twitter:card">
<meta content="AI/DDQN.py · 5GM-WP4 · Sina Khatibi / SDNController" property="twitter:title">
<meta content="GitLab.com" property="twitter:description">
<meta content="https://assets.gitlab-static.net/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="twitter:image">

<title>AI/DDQN.py · 5GM-WP4 · Sina Khatibi / SDNController · GitLab</title>
<meta content="GitLab.com" name="description">
<link rel="shortcut icon" type="image/png" href="https://gitlab.com/assets/favicon-7901bd695fb93edb07975966062049829afb56cf11511236e61bcf425070e36e.png" id="favicon" data-original-href="https://gitlab.com/assets/favicon-7901bd695fb93edb07975966062049829afb56cf11511236e61bcf425070e36e.png" />
<link rel="stylesheet" media="all" href="https://assets.gitlab-static.net/assets/application-318ee33e5d14035b04832fa07c492cdf57788adda50bb5219ef75b735cbf00e2.css" />
<link rel="stylesheet" media="print" href="https://assets.gitlab-static.net/assets/print-74c3df10dad473d66660c828e3aa54ca3bfeac6d8bb708643331403fe7211e60.css" />



<link rel="stylesheet" media="all" href="https://assets.gitlab-static.net/assets/highlight/themes/dark-9ed1f5e0afc6c7729fe361f5cfb4e9fae2bc416bc0fb7daae931c3f970bcb451.css" />
<script>
//<![CDATA[
window.gon={};gon.api_version="v4";gon.default_avatar_url="https://assets.gitlab-static.net/assets/no_avatar-849f9c04a3a0d0cea2424ae97b27447dc64a7dbfae83c036c45b403392f0e8ba.png";gon.max_file_size=10;gon.asset_host="https://assets.gitlab-static.net";gon.webpack_public_path="https://assets.gitlab-static.net/assets/webpack/";gon.relative_url_root="";gon.shortcuts_path="/help/shortcuts";gon.user_color_scheme="dark";gon.sentry_dsn="https://526a2f38a53d44e3a8e69bfa001d1e8b@sentry.gitlab.net/15";gon.sentry_environment=null;gon.gitlab_url="https://gitlab.com";gon.revision="cfdecb7c5de";gon.gitlab_logo="https://assets.gitlab-static.net/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png";gon.sprite_icons="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg";gon.sprite_file_icons="https://gitlab.com/assets/file_icons-7262fc6897e02f1ceaf8de43dc33afa5e4f9a2067f4f68ef77dcc87946575e9e.svg";gon.emoji_sprites_css_path="https://assets.gitlab-static.net/assets/emoji_sprites-289eccffb1183c188b630297431be837765d9ff4aed6130cf738586fb307c170.css";gon.test_env=false;gon.suggested_label_colors=["#0033CC","#428BCA","#44AD8E","#A8D695","#5CB85C","#69D100","#004E00","#34495E","#7F8C8D","#A295D6","#5843AD","#8E44AD","#FFECDB","#AD4363","#D10069","#CC0033","#FF0000","#D9534F","#D1D100","#F0AD4E","#AD8D43"];gon.first_day_of_week=0;gon.ee=true;gon.current_user_id=1235538;gon.current_username="Sinakh";gon.current_user_fullname="Sina Khatibi";gon.current_user_avatar_url="/uploads/-/system/user/avatar/1235538/avatar.png";
//]]>
</script>

<script src="https://assets.gitlab-static.net/assets/webpack/runtime.901bebd8.bundle.js" defer="defer"></script>
<script src="https://assets.gitlab-static.net/assets/webpack/main.8521c640.chunk.js" defer="defer"></script>
<script src="https://assets.gitlab-static.net/assets/webpack/raven.839fc8bd.chunk.js" defer="defer"></script>
<script src="https://assets.gitlab-static.net/assets/webpack/commons~pages.admin.clusters~pages.admin.clusters.destroy~pages.admin.clusters.edit~pages.admin.clus~a2ef139c.04bf4fba.chunk.js" defer="defer"></script>
<script src="https://assets.gitlab-static.net/assets/webpack/commons~pages.groups.epics.index~pages.groups.epics.show~pages.groups.milestones.edit~pages.groups.m~14875979.a05c0e92.chunk.js" defer="defer"></script>
<script src="https://assets.gitlab-static.net/assets/webpack/pages.projects.blob.show.ee03c3bb.chunk.js" defer="defer"></script>
<script>
  window.uploads_path = "/Sinakh/SDNController/uploads";
</script>

<meta name="csrf-param" content="authenticity_token" />
<meta name="csrf-token" content="kWzl78ma73VuG7SiPfv26eQKZtQhfp5m37tth5oTYkWSo0n8l74MwYlEdmGvM/Zt+11gTNRSb5z3dQCQ3vYjSA==" />
<meta content="origin-when-cross-origin" name="referrer">
<meta content="width=device-width, initial-scale=1, maximum-scale=1" name="viewport">
<meta content="#474D57" name="theme-color">
<link rel="apple-touch-icon" type="image/x-icon" href="https://assets.gitlab-static.net/assets/touch-icon-iphone-5a9cee0e8a51212e70b90c87c12f382c428870c0ff67d1eb034d884b78d2dae7.png" />
<link rel="apple-touch-icon" type="image/x-icon" href="https://assets.gitlab-static.net/assets/touch-icon-ipad-a6eec6aeb9da138e507593b464fdac213047e49d3093fc30e90d9a995df83ba3.png" sizes="76x76" />
<link rel="apple-touch-icon" type="image/x-icon" href="https://assets.gitlab-static.net/assets/touch-icon-iphone-retina-72e2aadf86513a56e050e7f0f2355deaa19cc17ed97bbe5147847f2748e5a3e3.png" sizes="120x120" />
<link rel="apple-touch-icon" type="image/x-icon" href="https://assets.gitlab-static.net/assets/touch-icon-ipad-retina-8ebe416f5313483d9c1bc772b5bbe03ecad52a54eba443e5215a22caed2a16a2.png" sizes="152x152" />
<link color="rgb(226, 67, 41)" href="https://assets.gitlab-static.net/assets/logo-d36b5212042cebc89b96df4bf6ac24e43db316143e89926c0db839ff694d2de4.svg" rel="mask-icon">
<meta content="https://assets.gitlab-static.net/assets/msapplication-tile-1196ec67452f618d39cdd85e2e3a542f76574c071051ae7effbfde01710eb17d.png" name="msapplication-TileImage">
<meta content="#30353E" name="msapplication-TileColor">



<script>
  ;(function(p,l,o,w,i,n,g){if(!p[i]){p.GlobalSnowplowNamespace=p.GlobalSnowplowNamespace||[];
  p.GlobalSnowplowNamespace.push(i);p[i]=function(){(p[i].q=p[i].q||[]).push(arguments)
  };p[i].q=p[i].q||[];n=l.createElement(o);g=l.getElementsByTagName(o)[0];n.async=1;
  n.src=w;g.parentNode.insertBefore(n,g)}}(window,document,"script","https://assets.gitlab-static.net/assets/snowplow/sp-e10fd598642f1a4dd3e9e0e026f6a1ffa3c31b8a40efd92db3f92d32873baed6.js","snowplow"));
  
  window.snowplow('newTracker', 'cf', 'snowplow.trx.gitlab.net', {
    appId: 'gitlab',
    cookieDomain: '.gitlab.com',
    userFingerprint: false,
    respectDoNotTrack: true,
    forceSecureTracker: true,
    post: true,
    contexts: {
        webPage: true,
    },
    stateStorageStrategy: "localStorage"
  });
  
  window.snowplow('enableActivityTracking', 30, 30);
  window.snowplow('trackPageView');
</script>


</head>

<body class="ui-indigo  gl-browser-chrome gl-platform-windows" data-find-file="/Sinakh/SDNController/find_file/5GM-WP4" data-group="" data-page="projects:blob:show" data-project="SDNController">

<script>
  gl = window.gl || {};
  gl.client = {"isChrome":true,"isWindows":true};
</script>



<header class="navbar navbar-gitlab qa-navbar navbar-expand-sm js-navbar">
<a class="sr-only gl-accessibility" href="#content-body" tabindex="1">Skip to content</a>
<div class="container-fluid">
<div class="header-content">
<div class="title-container">
<h1 class="title">
<a title="Dashboard" id="logo" href="/"><svg width="24" height="24" class="tanuki-logo" viewBox="0 0 36 36">
  <path class="tanuki-shape tanuki-left-ear" fill="#e24329" d="M2 14l9.38 9v-9l-4-12.28c-.205-.632-1.176-.632-1.38 0z"/>
  <path class="tanuki-shape tanuki-right-ear" fill="#e24329" d="M34 14l-9.38 9v-9l4-12.28c.205-.632 1.176-.632 1.38 0z"/>
  <path class="tanuki-shape tanuki-nose" fill="#e24329" d="M18,34.38 3,14 33,14 Z"/>
  <path class="tanuki-shape tanuki-left-eye" fill="#fc6d26" d="M18,34.38 11.38,14 2,14 6,25Z"/>
  <path class="tanuki-shape tanuki-right-eye" fill="#fc6d26" d="M18,34.38 24.62,14 34,14 30,25Z"/>
  <path class="tanuki-shape tanuki-left-cheek" fill="#fca326" d="M2 14L.1 20.16c-.18.565 0 1.2.5 1.56l17.42 12.66z"/>
  <path class="tanuki-shape tanuki-right-cheek" fill="#fca326" d="M34 14l1.9 6.16c.18.565 0 1.2-.5 1.56L18 34.38z"/>
</svg>

<span class="logo-text d-none d-lg-block prepend-left-8">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 617 169"><path d="M315.26 2.97h-21.8l.1 162.5h88.3v-20.1h-66.5l-.1-142.4M465.89 136.95c-5.5 5.7-14.6 11.4-27 11.4-16.6 0-23.3-8.2-23.3-18.9 0-16.1 11.2-23.8 35-23.8 4.5 0 11.7.5 15.4 1.2v30.1h-.1m-22.6-98.5c-17.6 0-33.8 6.2-46.4 16.7l7.7 13.4c8.9-5.2 19.8-10.4 35.5-10.4 17.9 0 25.8 9.2 25.8 24.6v7.9c-3.5-.7-10.7-1.2-15.1-1.2-38.2 0-57.6 13.4-57.6 41.4 0 25.1 15.4 37.7 38.7 37.7 15.7 0 30.8-7.2 36-18.9l4 15.9h15.4v-83.2c-.1-26.3-11.5-43.9-44-43.9M557.63 149.1c-8.2 0-15.4-1-20.8-3.5V70.5c7.4-6.2 16.6-10.7 28.3-10.7 21.1 0 29.2 14.9 29.2 39 0 34.2-13.1 50.3-36.7 50.3m9.2-110.6c-19.5 0-30 13.3-30 13.3v-21l-.1-27.8h-21.3l.1 158.5c10.7 4.5 25.3 6.9 41.2 6.9 40.7 0 60.3-26 60.3-70.9-.1-35.5-18.2-59-50.2-59M77.9 20.6c19.3 0 31.8 6.4 39.9 12.9l9.4-16.3C114.5 6 97.3 0 78.9 0 32.5 0 0 28.3 0 85.4c0 59.8 35.1 83.1 75.2 83.1 20.1 0 37.2-4.7 48.4-9.4l-.5-63.9V75.1H63.6v20.1h38l.5 48.5c-5 2.5-13.6 4.5-25.3 4.5-32.2 0-53.8-20.3-53.8-63-.1-43.5 22.2-64.6 54.9-64.6M231.43 2.95h-21.3l.1 27.3v94.3c0 26.3 11.4 43.9 43.9 43.9 4.5 0 8.9-.4 13.1-1.2v-19.1c-3.1.5-6.4.7-9.9.7-17.9 0-25.8-9.2-25.8-24.6v-65h35.7v-17.8h-35.7l-.1-38.5M155.96 165.47h21.3v-124h-21.3v124M155.96 24.37h21.3V3.07h-21.3v21.3"/></svg>

</span>
</a><a class="label-link js-canary-badge canary-badge bg-transparent hidden" target="_blank" href="https://next.gitlab.com"><span class="color-label has-tooltip badge badge-pill green-badge">
Next
</span>
</a></h1>
<ul class="list-unstyled navbar-sub-nav">
<li id="nav-projects-dropdown" class="home dropdown header-projects qa-projects-dropdown" data-track-label="projects_dropdown" data-track-event="click_dropdown"><button class="btn" data-toggle="dropdown" type="button">
Projects
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</button>
<div class="dropdown-menu frequent-items-dropdown-menu">
<div class="frequent-items-dropdown-container">
<div class="frequent-items-dropdown-sidebar qa-projects-dropdown-sidebar">
<ul>
<li class=""><a class="qa-your-projects-link" href="/dashboard/projects">Your projects
</a></li><li class=""><a href="/dashboard/projects/starred">Starred projects
</a></li><li class=""><a href="/explore">Explore projects
</a></li></ul>
</div>
<div class="frequent-items-dropdown-content">
<div data-project-id="6432512" data-project-name="SDNController" data-project-namespace="Sina Khatibi / SDNController" data-project-web-url="/Sinakh/SDNController" data-user-name="Sinakh" id="js-projects-dropdown"></div>
</div>
</div>

</div>
</li><li id="nav-groups-dropdown" class="home dropdown header-groups qa-groups-dropdown" data-track-label="groups_dropdown" data-track-event="click_dropdown"><button class="btn" data-toggle="dropdown" type="button">
Groups
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</button>
<div class="dropdown-menu frequent-items-dropdown-menu">
<div class="frequent-items-dropdown-container">
<div class="frequent-items-dropdown-sidebar qa-groups-dropdown-sidebar">
<ul>
<li class=""><a class="qa-your-groups-link" href="/dashboard/groups">Your groups
</a></li><li class=""><a href="/explore/groups">Explore groups
</a></li></ul>
</div>
<div class="frequent-items-dropdown-content">
<div data-user-name="Sinakh" id="js-groups-dropdown"></div>
</div>
</div>

</div>
</li><li class="d-none d-xl-block d-lg-block"><a class="dashboard-shortcuts-activity" title="Activity" href="/dashboard/activity">Activity
</a></li><li class="d-none d-xl-block d-lg-block"><a class="dashboard-shortcuts-milestones" title="Milestones" href="/dashboard/milestones">Milestones
</a></li><li class="d-none d-xl-block d-lg-block"><a class="dashboard-shortcuts-snippets qa-snippets-link" title="Snippets" href="/dashboard/snippets">Snippets
</a></li><li class="d-lg-none d-xl-none dropdown header-more">
<a data-toggle="dropdown" href="#">
More
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</a>
<div class="dropdown-menu">
<ul>
<li class=""><a title="Activity" href="/dashboard/activity">Activity
</a></li><li class=""><a class="dashboard-shortcuts-milestones" title="Milestones" href="/dashboard/milestones">Milestones
</a></li><li class=""><a class="dashboard-shortcuts-snippets" title="Snippets" href="/dashboard/snippets">Snippets
</a></li><li class="dropdown">
<button aria-expanded="false" aria-haspopup="true" aria-label="Operations Dashboard" class="btn-link" data-toggle="dropdown" id="js-dashboards-menu" type="button">
<svg class="s18"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#dashboard"></use></svg>
</button>
<div aria-labelledby="js-dashboards-menu" class="dropdown-menu">
<div class="dropdown-bold-header">
Dashboards
</div>
<a class="dropdown-item" title="Operations" aria-label="Operations" href="/-/operations">Operations
</a></div>
</li>

</ul>
</div>
</li>
<li class="hidden">
<a title="Projects" class="dashboard-shortcuts-projects" href="/dashboard/projects">Projects
</a></li>
<li class="dropdown">
<button aria-expanded="false" aria-haspopup="true" aria-label="Operations Dashboard" class="btn-link" data-toggle="dropdown" id="js-dashboards-menu" type="button">
<svg class="s18"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#dashboard"></use></svg>
</button>
<div aria-labelledby="js-dashboards-menu" class="dropdown-menu">
<div class="dropdown-bold-header">
Dashboards
</div>
<a class="dropdown-item" title="Operations" aria-label="Operations" href="/-/operations">Operations
</a></div>
</li>

</ul>

</div>
<div class="navbar-collapse collapse">
<ul class="nav navbar-nav">
<li class="header-new dropdown" data-track-event="click_dropdown" data-track-label="new_dropdown">
<a class="header-new-dropdown-toggle has-tooltip qa-new-menu-toggle" title="New..." ref="tooltip" aria-label="New..." data-toggle="dropdown" data-placement="bottom" data-container="body" data-display="static" href="/projects/new"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#plus-square"></use></svg>
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</a><div class="dropdown-menu dropdown-menu-right">
<ul>
<li class="dropdown-bold-header">
This project
</li>
<li><a href="/Sinakh/SDNController/issues/new">New issue</a></li>
<li><a href="/Sinakh/SDNController/merge_requests/new">New merge request</a></li>
<li><a href="/Sinakh/SDNController/snippets/new">New snippet</a></li>
<li class="divider"></li>
<li class="dropdown-bold-header">GitLab</li>
<li><a class="qa-global-new-project-link" href="/projects/new">New project</a></li>
<li><a href="/groups/new">New group</a></li>
<li><a class="qa-global-new-snippet-link" href="/snippets/new">New snippet</a></li>
</ul>
</div>
</li>

<li class="nav-item d-none d-sm-none d-md-block m-auto">
<div class="search search-form" data-track-event="activate_form_input" data-track-label="navbar_search">
<form class="form-inline" action="/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><div class="search-input-container">
<div class="search-input-wrap">
<div class="dropdown" data-url="/search/autocomplete">
<input type="search" name="search" id="search" placeholder="Search or jump to…" class="search-input dropdown-menu-toggle no-outline js-search-dashboard-options" spellcheck="false" tabindex="1" autocomplete="off" data-issues-path="/dashboard/issues" data-mr-path="/dashboard/merge_requests" aria-label="Search or jump to…" />
<button class="hidden js-dropdown-search-toggle" data-toggle="dropdown" type="button"></button>
<div class="dropdown-menu dropdown-select">
<div class="dropdown-content"><ul>
<li class="dropdown-menu-empty-item">
<a>
Loading...
</a>
</li>
</ul>
</div><div class="dropdown-loading"><i aria-hidden="true" data-hidden="true" class="fa fa-spinner fa-spin"></i></div>
</div>
<svg class="s16 search-icon"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#search"></use></svg>
<svg class="s16 clear-icon js-clear-input"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#close"></use></svg>
</div>
</div>
</div>
<input type="hidden" name="group_id" id="group_id" class="js-search-group-options" />
<input type="hidden" name="project_id" id="search_project_id" value="6432512" class="js-search-project-options" data-project-path="SDNController" data-name="SDNController" data-issues-path="/Sinakh/SDNController/issues" data-mr-path="/Sinakh/SDNController/merge_requests" data-issues-disabled="false" />
<input type="hidden" name="search_code" id="search_code" value="true" />
<input type="hidden" name="repository_ref" id="repository_ref" value="5GM-WP4" />

<div class="search-autocomplete-opts hide" data-autocomplete-path="/search/autocomplete" data-autocomplete-project-id="6432512" data-autocomplete-project-ref="5GM-WP4"></div>
</form></div>

</li>
<li class="nav-item d-inline-block d-sm-none d-md-none">
<a title="Search" aria-label="Search" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/search?project_id=6432512"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#search"></use></svg>
</a></li>
<li class="user-counter"><a title="Issues" class="dashboard-shortcuts-issues" aria-label="Issues" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/issues?assignee_username=Sinakh"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#issues"></use></svg>
<span class="badge badge-pill green-badge hidden issues-count">
0
</span>
</a></li><li class="user-counter"><a title="Merge requests" class="dashboard-shortcuts-merge_requests" aria-label="Merge requests" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/merge_requests?assignee_username=Sinakh"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#git-merge"></use></svg>
<span class="badge badge-pill hidden merge-requests-count">
0
</span>
</a></li><li class="user-counter"><a title="Todos" aria-label="Todos" class="shortcuts-todos" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/todos"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#todo-done"></use></svg>
<span class="badge badge-pill hidden todos-count">
0
</span>
</a></li><li class="nav-item header-help dropdown">
<a class="header-help-dropdown-toggle" data-toggle="dropdown" href="/help"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#question"></use></svg>
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</a><div class="dropdown-menu dropdown-menu-right">
<ul>
<li>
<a href="/help">Help</a>
</li>

<li class="divider"></li>
<li>
<a href="https://about.gitlab.com/submit-feedback">Submit feedback</a>
</li>
<li>
<a target="_blank" class="text-nowrap" href="https://about.gitlab.com/contributing">Contribute to GitLab
</a></li>


<li class="js-canary-link">
<a href="https://next.gitlab.com/">Switch to GitLab Next</a>
</li>
</ul>

</div>
</li>
<li class="nav-item header-user dropdown" data-track-event="click_dropdown" data-track-label="profile_dropdown">
<a class="header-user-dropdown-toggle" data-toggle="dropdown" href="/Sinakh"><img width="23" height="23" class="header-user-avatar qa-user-avatar lazy" data-src="https://assets.gitlab-static.net/uploads/-/system/user/avatar/1235538/avatar.png?width=23" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" />
<svg class="caret-down"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-down"></use></svg>
</a><div class="dropdown-menu dropdown-menu-right">
<ul>
<li class="current-user">
<div class="user-name bold">
Sina Khatibi
</div>
@Sinakh
</li>
<li class="divider"></li>
<li>
<div class="js-set-status-modal-trigger" data-has-status="false"></div>
</li>
<li>
<a class="profile-link" data-user="Sinakh" href="/Sinakh">Profile</a>
</li>
<li>
<a href="/profile">Settings</a>
</li>
<li class="divider"></li>
<li>
<a class="sign-out-link" href="/users/sign_out">Sign out</a>
</li>
</ul>

</div>
</li>
</ul>
</div>
<button class="navbar-toggler d-block d-sm-none" type="button">
<span class="sr-only">Toggle navigation</span>
<svg class="s12 more-icon js-navbar-toggle-right"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#ellipsis_h"></use></svg>
<svg class="s12 close-icon js-navbar-toggle-left"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#close"></use></svg>
</button>
</div>
</div>
</header>
<div class="js-set-status-modal-wrapper" data-current-emoji="" data-current-message=""></div>

<div class="layout-page page-with-contextual-sidebar">
<div class="nav-sidebar">
<div class="nav-sidebar-inner-scroll">
<div class="context-header">
<a title="SDNController" href="/Sinakh/SDNController"><div class="avatar-container rect-avatar s40 project-avatar">
<div class="avatar s40 avatar-tile identicon bg3">S</div>
</div>
<div class="sidebar-context-title">
SDNController
</div>
</a></div>
<ul class="sidebar-top-level-items">
<li class="home"><a class="shortcuts-project" href="/Sinakh/SDNController"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#home"></use></svg>
</div>
<span class="nav-item-name">
Project
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController"><strong class="fly-out-top-item-name">
Project
</strong>
</a></li><li class="divider fly-out-top-item"></li>
<li class=""><a title="Project details" class="shortcuts-project" href="/Sinakh/SDNController"><span>Details</span>
</a></li><li class=""><a title="Activity" class="shortcuts-project-activity qa-activity-link" href="/Sinakh/SDNController/activity"><span>Activity</span>
</a></li><li class=""><a title="Releases" class="shortcuts-project-releases" href="/Sinakh/SDNController/releases"><span>Releases</span>
</a></li>
<li class=""><a title="Cycle Analytics" class="shortcuts-project-cycle-analytics" href="/Sinakh/SDNController/cycle_analytics"><span>Cycle Analytics</span>
</a></li>
</ul>
</li><li class="active"><a class="shortcuts-tree qa-project-menu-repo" href="/Sinakh/SDNController/tree/5GM-WP4"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#doc-text"></use></svg>
</div>
<span class="nav-item-name">
Repository
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item active"><a href="/Sinakh/SDNController/tree/5GM-WP4"><strong class="fly-out-top-item-name">
Repository
</strong>
</a></li><li class="divider fly-out-top-item"></li>
<li class="active"><a href="/Sinakh/SDNController/tree/5GM-WP4">Files
</a></li><li class=""><a href="/Sinakh/SDNController/commits/5GM-WP4">Commits
</a></li><li class=""><a class="qa-branches-link" href="/Sinakh/SDNController/branches">Branches
</a></li><li class=""><a href="/Sinakh/SDNController/tags">Tags
</a></li><li class=""><a href="/Sinakh/SDNController/graphs/5GM-WP4">Contributors
</a></li><li class=""><a href="/Sinakh/SDNController/network/5GM-WP4">Graph
</a></li><li class=""><a href="/Sinakh/SDNController/compare?from=master&amp;to=5GM-WP4">Compare
</a></li><li class=""><a href="/Sinakh/SDNController/graphs/5GM-WP4/charts">Charts
</a></li>
</ul>
</li><li class=""><a class="shortcuts-issues qa-issues-item" href="/Sinakh/SDNController/issues"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#issues"></use></svg>
</div>
<span class="nav-item-name">
Issues
</span>
<span class="badge badge-pill count issue_counter">
0
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/issues"><strong class="fly-out-top-item-name">
Issues
</strong>
<span class="badge badge-pill count issue_counter fly-out-badge">
0
</span>
</a></li><li class="divider fly-out-top-item"></li>
<li class=""><a title="Issues" href="/Sinakh/SDNController/issues"><span>
List
</span>
</a></li><li class=""><a title="Board" href="/Sinakh/SDNController/boards"><span>
Board
</span>
</a></li><li class=""><a title="Labels" class="qa-labels-link" href="/Sinakh/SDNController/labels"><span>
Labels
</span>
</a></li>
<li class=""><a title="Milestones" class="qa-milestones-link" href="/Sinakh/SDNController/milestones"><span>
Milestones
</span>
</a></li></ul>
</li><li class=""><a class="shortcuts-merge_requests qa-merge-requests-link" href="/Sinakh/SDNController/merge_requests"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#git-merge"></use></svg>
</div>
<span class="nav-item-name">
Merge Requests
</span>
<span class="badge badge-pill count merge_counter js-merge-counter">
0
</span>
</a><ul class="sidebar-sub-level-items is-fly-out-only">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/merge_requests"><strong class="fly-out-top-item-name">
Merge Requests
</strong>
<span class="badge badge-pill count merge_counter js-merge-counter fly-out-badge">
0
</span>
</a></li></ul>
</li><li class=""><a class="shortcuts-pipelines qa-link-pipelines" href="/Sinakh/SDNController/pipelines"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#rocket"></use></svg>
</div>
<span class="nav-item-name">
CI / CD
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/pipelines"><strong class="fly-out-top-item-name">
CI / CD
</strong>
</a></li><li class="divider fly-out-top-item"></li>
<li class=""><a title="Pipelines" class="shortcuts-pipelines" href="/Sinakh/SDNController/pipelines"><span>
Pipelines
</span>
</a></li><li class=""><a title="Jobs" class="shortcuts-builds" href="/Sinakh/SDNController/-/jobs"><span>
Jobs
</span>
</a></li><li class=""><a title="Schedules" class="shortcuts-builds" href="/Sinakh/SDNController/pipeline_schedules"><span>
Schedules
</span>
</a></li><li class=""><a title="Charts" class="shortcuts-pipelines-charts" href="/Sinakh/SDNController/pipelines/charts"><span>
Charts
</span>
</a></li></ul>
</li><li class=""><a class="shortcuts-operations qa-link-operations" href="/Sinakh/SDNController/environments/metrics"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#cloud-gear"></use></svg>
</div>
<span class="nav-item-name">
Operations
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/environments/metrics"><strong class="fly-out-top-item-name">
Operations
</strong>
</a></li><li class="divider fly-out-top-item"></li>
<li class=""><a title="Metrics" class="shortcuts-metrics" href="/Sinakh/SDNController/environments/metrics"><span>
Metrics
</span>
</a></li>
<li class=""><a title="Environments" class="shortcuts-environments qa-operations-environments-link" href="/Sinakh/SDNController/environments"><span>
Environments
</span>
</a></li><li class=""><a title="Error Tracking" class="shortcuts-tracking qa-operations-tracking-link" href="/Sinakh/SDNController/error_tracking"><span>
Error Tracking
</span>
</a></li><li class=""><a title="Serverless" href="/Sinakh/SDNController/serverless/functions"><span>
Serverless
</span>
</a></li><li class=""><a title="Kubernetes" class="shortcuts-kubernetes" href="/Sinakh/SDNController/clusters"><span>
Kubernetes
</span>
<div class="feature-highlight js-feature-highlight" data-container="body" data-dismiss-endpoint="/-/user_callouts" data-highlight="gke_cluster_integration" data-placement="right" data-toggle="popover" data-trigger="manual" disabled></div>
</a><div class="feature-highlight-popover-content">
<img class="feature-highlight-illustration lazy" data-src="https://assets.gitlab-static.net/assets/illustrations/cluster_popover-9830388038d966d8d64d43576808f9d5ba05f639a78a40bae9a5ddc7cbf72f24.svg" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" />
<div class="feature-highlight-popover-sub-content">
<p>Allows you to add and manage Kubernetes clusters.</p>
<p>
Protip:
<a href="/help/topics/autodevops/index.md">Auto DevOps</a>
<span>uses Kubernetes clusters to deploy your code!</span>
</p>
<hr>
<button class="btn btn-success btn-sm dismiss-feature-highlight" type="button">
<span>Got it!</span>
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#thumb-up"></use></svg>
</button>
</div>
</div>
</li></ul>
</li><li class=""><a class="shortcuts-container-registry" href="/Sinakh/SDNController/container_registry"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#disk"></use></svg>
</div>
<span class="nav-item-name">
Registry
</span>
</a><ul class="sidebar-sub-level-items is-fly-out-only">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/container_registry"><strong class="fly-out-top-item-name">
Registry
</strong>
</a></li></ul>
</li><li class=""><a class="shortcuts-wiki qa-wiki-link" href="/Sinakh/SDNController/wikis/home"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#book"></use></svg>
</div>
<span class="nav-item-name">
Wiki
</span>
</a><ul class="sidebar-sub-level-items is-fly-out-only">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/wikis/home"><strong class="fly-out-top-item-name">
Wiki
</strong>
</a></li></ul>
</li><li class=""><a class="shortcuts-snippets" href="/Sinakh/SDNController/snippets"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#snippet"></use></svg>
</div>
<span class="nav-item-name">
Snippets
</span>
</a><ul class="sidebar-sub-level-items is-fly-out-only">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/snippets"><strong class="fly-out-top-item-name">
Snippets
</strong>
</a></li></ul>
</li><li class=""><a class="shortcuts-tree" href="/Sinakh/SDNController/edit"><div class="nav-icon-container">
<svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#settings"></use></svg>
</div>
<span class="nav-item-name qa-settings-item">
Settings
</span>
</a><ul class="sidebar-sub-level-items">
<li class="fly-out-top-item"><a href="/Sinakh/SDNController/edit"><strong class="fly-out-top-item-name">
Settings
</strong>
</a></li><li class="divider fly-out-top-item"></li>
<li class=""><a title="General" href="/Sinakh/SDNController/edit"><span>
General
</span>
</a></li><li class=""><a title="Members" class="qa-link-members-settings" href="/Sinakh/SDNController/project_members"><span>
Members
</span>
</a></li><li class=""><a title="Integrations" href="/Sinakh/SDNController/settings/integrations"><span>
Integrations
</span>
</a></li><li class=""><a title="Repository" href="/Sinakh/SDNController/settings/repository"><span>
Repository
</span>
</a></li><li class=""><a title="CI / CD" href="/Sinakh/SDNController/settings/ci_cd"><span>
CI / CD
</span>
</a></li><li class=""><a title="Operations" href="/Sinakh/SDNController/settings/operations">Operations
</a></li><li class=""><a title="Pages" href="/Sinakh/SDNController/pages"><span>
Pages
</span>
</a></li><li class=""><a title="Audit Events" href="/Sinakh/SDNController/audit_events">Audit Events
</a></li>
</ul>
</li><a class="toggle-sidebar-button js-toggle-sidebar" role="button" title="Toggle sidebar" type="button">
<svg class="icon-angle-double-left"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-double-left"></use></svg>
<svg class="icon-angle-double-right"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-double-right"></use></svg>
<span class="collapse-text">Collapse sidebar</span>
</a>
<button name="button" type="button" class="close-nav-button"><svg class="s16"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#close"></use></svg>
<span class="collapse-text">Close sidebar</span>
</button>
<li class="hidden">
<a title="Activity" class="shortcuts-project-activity" href="/Sinakh/SDNController/activity"><span>
Activity
</span>
</a></li>
<li class="hidden">
<a title="Network" class="shortcuts-network" href="/Sinakh/SDNController/network/5GM-WP4">Graph
</a></li>
<li class="hidden">
<a title="Charts" class="shortcuts-repository-charts" href="/Sinakh/SDNController/graphs/5GM-WP4/charts">Charts
</a></li>
<li class="hidden">
<a class="shortcuts-new-issue" href="/Sinakh/SDNController/issues/new">Create a new issue
</a></li>
<li class="hidden">
<a title="Jobs" class="shortcuts-builds" href="/Sinakh/SDNController/-/jobs">Jobs
</a></li>
<li class="hidden">
<a title="Commits" class="shortcuts-commits" href="/Sinakh/SDNController/commits/5GM-WP4">Commits
</a></li>
<li class="hidden">
<a title="Issue Boards" class="shortcuts-issue-boards" href="/Sinakh/SDNController/boards">Issue Boards</a>
</li>
</ul>
</div>
</div>

<div class="content-wrapper">

<div class="mobile-overlay"></div>
<div class="alert-wrapper">






<nav class="breadcrumbs container-fluid container-limited" role="navigation">
<div class="breadcrumbs-container">
<button name="button" type="button" class="toggle-mobile-nav"><span class="sr-only">Open sidebar</span>
<i aria-hidden="true" data-hidden="true" class="fa fa-bars"></i>
</button><div class="breadcrumbs-links js-title-container">
<ul class="list-unstyled breadcrumbs-list js-breadcrumbs-list">
<li><a href="/Sinakh">Sina Khatibi</a><svg class="s8 breadcrumbs-list-angle"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-right"></use></svg></li> <li><a href="/Sinakh/SDNController"><span class="breadcrumb-item-text js-breadcrumb-item-text">SDNController</span></a><svg class="s8 breadcrumbs-list-angle"><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#angle-right"></use></svg></li>

<li>
<h2 class="breadcrumbs-sub-title"><a href="/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py">Repository</a></h2>
</li>
</ul>
</div>

</div>
</nav>

<div class="flash-container flash-container-page">
</div>

<div class="d-flex"></div>
</div>
<div class=" ">
<div class="content" id="content-body">
<div class="js-signature-container" data-signatures-path="/Sinakh/SDNController/commits/2e3d61317693315da31aab0dff9fed519e2db861/signatures"></div>
<div class="container-fluid container-limited">

<div class="tree-holder" id="tree-holder">
<div class="nav-block">
<div class="tree-ref-container">
<div class="tree-ref-holder">
<form class="project-refs-form" action="/Sinakh/SDNController/refs/switch" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="destination" id="destination" value="blob" />
<input type="hidden" name="path" id="path" value="AI/DDQN.py" />
<div class="dropdown">
<button class="dropdown-menu-toggle js-project-refs-dropdown qa-branches-select" type="button" data-toggle="dropdown" data-selected="5GM-WP4" data-ref="5GM-WP4" data-refs-url="/Sinakh/SDNController/refs?sort=updated_desc" data-field-name="ref" data-submit-form-on-click="true" data-visit="true"><span class="dropdown-toggle-text ">5GM-WP4</span><i aria-hidden="true" data-hidden="true" class="fa fa-chevron-down"></i></button>
<div class="dropdown-menu dropdown-menu-paging dropdown-menu-selectable git-revision-dropdown qa-branches-dropdown">
<div class="dropdown-page-one">
<div class="dropdown-title"><span>Switch branch/tag</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i aria-hidden="true" data-hidden="true" class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search branches and tags" autocomplete="off" /><i aria-hidden="true" data-hidden="true" class="fa fa-search dropdown-input-search"></i><i aria-hidden="true" data-hidden="true" role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i aria-hidden="true" data-hidden="true" class="fa fa-spinner fa-spin"></i></div>
</div>
</div>
</div>
</form>
</div>
<ul class="breadcrumb repo-breadcrumb">
<li class="breadcrumb-item">
<a href="/Sinakh/SDNController/tree/5GM-WP4">SDNController
</a></li>
<li class="breadcrumb-item">
<a href="/Sinakh/SDNController/tree/5GM-WP4/AI">AI</a>
</li>
<li class="breadcrumb-item">
<a href="/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py"><strong>DDQN.py</strong>
</a></li>
</ul>
</div>
<div class="tree-controls">
<a class="btn shortcuts-find-file" rel="nofollow" href="/Sinakh/SDNController/find_file/5GM-WP4"><i aria-hidden="true" data-hidden="true" class="fa fa-search"></i>
<span>Find file</span>
</a>
<div class="btn-group" role="group"><a class="btn js-blob-blame-link" href="/Sinakh/SDNController/blame/5GM-WP4/AI/DDQN.py">Blame</a><a class="btn" href="/Sinakh/SDNController/commits/5GM-WP4/AI/DDQN.py">History</a><a class="btn js-data-file-blob-permalink-url" href="/Sinakh/SDNController/blob/50c2fc04fa9b0f8cbeabe1e282c64c1bd633e34b/AI/DDQN.py">Permalink</a></div>
</div>
</div>

<div class="info-well d-none d-sm-block">
<div class="well-segment">
<ul class="blob-commit-info">
<li class="commit flex-row js-toggle-container" id="commit-2e3d6131">
<div class="avatar-cell d-none d-sm-block">
<a href="mailto:khatibi@nomor.de"><img alt="Sina Khatibi&#39;s avatar" src="https://secure.gravatar.com/avatar/76c1ec112e8371b58e54a1b4818e584c?s=72&amp;d=identicon" class="avatar s36 d-none d-sm-inline" title="Sina Khatibi" /></a>
</div>
<div class="commit-detail flex-list">
<div class="commit-content qa-commit-content">
<a class="commit-row-message item-title" href="/Sinakh/SDNController/commit/2e3d61317693315da31aab0dff9fed519e2db861">DQN Class is finally working.</a>
<span class="commit-row-message d-inline d-sm-none">
&middot;
2e3d6131
</span>
<div class="committer">
<a class="commit-author-link" href="mailto:khatibi@nomor.de">Sina Khatibi</a> authored <time class="js-timeago" title="Nov 29, 2018 4:54pm" datetime="2018-11-29T16:54:17Z" data-toggle="tooltip" data-placement="bottom" data-container="body">Nov 29, 2018</time>
</div>
</div>
<div class="commit-actions flex-row">

<div class="js-commit-pipeline-status" data-endpoint="/Sinakh/SDNController/commit/2e3d61317693315da31aab0dff9fed519e2db861/pipelines?ref=5GM-WP4"></div>
<div class="commit-sha-group d-none d-sm-flex">
<div class="label label-monospace monospace">
2e3d6131
</div>
<button class="btn btn btn-default" data-toggle="tooltip" data-placement="bottom" data-container="body" data-title="Copy commit SHA to clipboard" data-class="btn btn-default" data-clipboard-text="2e3d61317693315da31aab0dff9fed519e2db861" type="button" title="Copy commit SHA to clipboard" aria-label="Copy commit SHA to clipboard"><svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#duplicate"></use></svg></button>

</div>
</div>
</div>
</li>

</ul>
</div>


</div>
<div class="blob-content-holder" id="blob-content-holder">
<article class="file-holder">
<div class="js-file-title file-title-flex-parent">
<div class="file-header-content">
<i aria-hidden="true" data-hidden="true" class="fa fa-file-text-o fa-fw"></i>
<strong class="file-title-name qa-file-title-name">
DDQN.py
</strong>
<button class="btn btn-clipboard btn-transparent prepend-left-5" data-toggle="tooltip" data-placement="bottom" data-container="body" data-class="btn-clipboard btn-transparent prepend-left-5" data-title="Copy file path to clipboard" data-clipboard-text="{&quot;text&quot;:&quot;AI/DDQN.py&quot;,&quot;gfm&quot;:&quot;`AI/DDQN.py`&quot;}" type="button" title="Copy file path to clipboard" aria-label="Copy file path to clipboard"><svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#duplicate"></use></svg></button>
<small>
9.82 KB
</small>
</div>

<div class="file-actions">

<div class="btn-group" role="group"><button class="btn btn btn-sm js-copy-blob-source-btn" data-toggle="tooltip" data-placement="bottom" data-container="body" data-class="btn btn-sm js-copy-blob-source-btn" data-title="Copy source to clipboard" data-clipboard-target=".blob-content[data-blob-id=&#39;d9a25a23b18e11bd0cd746e06ce65ba49482c6c2&#39;]" type="button" title="Copy source to clipboard" aria-label="Copy source to clipboard"><svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#duplicate"></use></svg></button><a class="btn btn-sm has-tooltip" target="_blank" rel="noopener noreferrer" title="Open raw" data-container="body" href="/Sinakh/SDNController/raw/5GM-WP4/AI/DDQN.py"><i aria-hidden="true" data-hidden="true" class="fa fa-file-code-o"></i></a><a download="AI/DDQN.py" class="btn btn-sm has-tooltip" target="_blank" rel="noopener noreferrer" title="Download" data-container="body" href="/Sinakh/SDNController/raw/5GM-WP4/AI/DDQN.py?inline=false"><svg><use xlink:href="https://gitlab.com/assets/icons-09fdf2c02921bad2ec7257465016a755f359ab7b598e5fe42c22381fe1a25045.svg#download"></use></svg></a></div>
<div class="btn-group" role="group">
<a class="btn js-edit-blob  btn-sm" href="/Sinakh/SDNController/edit/5GM-WP4/AI/DDQN.py">Edit</a><a class="btn btn-default btn-sm" href="/-/ide/project/Sinakh/SDNController/edit/5GM-WP4/-/AI/DDQN.py">Web IDE</a><button name="button" type="submit" class="btn btn-default" data-target="#modal-upload-blob" data-toggle="modal">Replace</button><button name="button" type="submit" class="btn btn-remove" data-target="#modal-remove-blob" data-toggle="modal">Delete</button></div>
</div>
</div>
<div class="js-file-fork-suggestion-section file-fork-suggestion hidden">
<span class="file-fork-suggestion-note">
You're not allowed to
<span class="js-file-fork-suggestion-section-action">
edit
</span>
files in this project directly. Please fork this project,
make your changes there, and submit a merge request.
</span>
<a class="js-fork-suggestion-button btn btn-grouped btn-inverted btn-success" rel="nofollow" data-method="post" href="/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py">Fork</a>
<button class="js-cancel-fork-suggestion-button btn btn-grouped" type="button">
Cancel
</button>
</div>



<div class="blob-viewer" data-type="simple" data-url="/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py?format=json&amp;viewer=simple">
<div class="text-center prepend-top-default append-bottom-default">
<i aria-hidden="true" aria-label="Loading content…" class="fa fa-spinner fa-spin fa-2x qa-spinner"></i>
</div>

</div>


</article>
</div>

<div class="modal" id="modal-remove-blob">
<div class="modal-dialog">
<div class="modal-content">
<div class="modal-header">
<h3 class="page-title">Delete DDQN.py</h3>
<button aria-label="Close" class="close" data-dismiss="modal" type="button">
<span aria-hidden="true">&times;</span>
</button>
</div>
<div class="modal-body">
<form class="js-delete-blob-form js-quick-submit js-requires-input" action="/Sinakh/SDNController/blob/5GM-WP4/AI/DDQN.py" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="delete" /><input type="hidden" name="authenticity_token" value="AqvNtIBWjZZpu7YDuI6WATBfoj5wMRo2UFoIGCPrBsgBZGGn3nJuIo7kdMAqRpaFLwikpoUd68x4lGUPZw5HxQ==" /><div class="form-group row commit_message-group">
<label class="col-form-label col-sm-2" for="commit_message-5ebc6737caa4a0557eb93e3437d04095">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-5ebc6737caa4a0557eb93e3437d04095" class="form-control js-commit-message" placeholder="Delete DDQN.py" required="required" rows="3">
Delete DDQN.py</textarea>
</div>
</div>
</div>

<div class="form-group row branch">
<label class="col-form-label col-sm-2" for="branch_name">Target Branch</label>
<div class="col-sm-10">
<input type="text" name="branch_name" id="branch_name" value="5GM-WP4" required="required" class="form-control js-branch-name ref-name" />
<div class="js-create-merge-request-container">
<div class="form-check prepend-top-8">
<input type="checkbox" name="create_merge_request" id="create_merge_request-21a2a61de89429437eac9e7c056722fd" value="1" class="js-create-merge-request form-check-input" checked="checked" />
<label class="form-check-label" for="create_merge_request-21a2a61de89429437eac9e7c056722fd">Start a <strong>new merge request</strong> with these changes
</label></div>

</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="5GM-WP4" class="js-original-branch" />

<div class="form-group row">
<div class="offset-sm-2 col-sm-10">
<button name="button" type="submit" class="btn btn-remove btn-remove-file">Delete file</button>
<a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>
</div>
</div>
</form></div>
</div>
</div>
</div>

<div class="modal" id="modal-upload-blob">
<div class="modal-dialog modal-lg">
<div class="modal-content">
<div class="modal-header">
<h3 class="page-title">Replace DDQN.py</h3>
<button aria-label="Close" class="close" data-dismiss="modal" type="button">
<span aria-hidden="true">&times;</span>
</button>
</div>
<div class="modal-body">
<form class="js-quick-submit js-upload-blob-form" data-method="put" action="/Sinakh/SDNController/update/5GM-WP4/AI/DDQN.py" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="mB+ER9q5nRE+w+lZzsaf/WMG8efkcFe2KMLj3Ha/kUSb0ChUhJ1+pdmcK5pcDp95fFH3fxFcpkwADI7LMlrQSQ==" /><div class="dropzone">
<div class="dropzone-previews blob-upload-dropzone-previews">
<p class="dz-message light">
Attach a file by drag &amp; drop or <a class="markdown-selector" href="#">click to upload</a>
</p>
</div>
</div>
<br>
<div class="dropzone-alerts alert alert-danger data" style="display:none"></div>
<div class="form-group row commit_message-group">
<label class="col-form-label col-sm-2" for="commit_message-711e923661678d58c5cedff808193f7c">Commit message
</label><div class="col-sm-10">
<div class="commit-message-container">
<div class="max-width-marker"></div>
<textarea name="commit_message" id="commit_message-711e923661678d58c5cedff808193f7c" class="form-control js-commit-message" placeholder="Replace DDQN.py" required="required" rows="3">
Replace DDQN.py</textarea>
</div>
</div>
</div>

<div class="form-group row branch">
<label class="col-form-label col-sm-2" for="branch_name">Target Branch</label>
<div class="col-sm-10">
<input type="text" name="branch_name" id="branch_name" value="5GM-WP4" required="required" class="form-control js-branch-name ref-name" />
<div class="js-create-merge-request-container">
<div class="form-check prepend-top-8">
<input type="checkbox" name="create_merge_request" id="create_merge_request-b2fdbdb13b453467a4ee5219117f94af" value="1" class="js-create-merge-request form-check-input" checked="checked" />
<label class="form-check-label" for="create_merge_request-b2fdbdb13b453467a4ee5219117f94af">Start a <strong>new merge request</strong> with these changes
</label></div>

</div>
</div>
</div>
<input type="hidden" name="original_branch" id="original_branch" value="5GM-WP4" class="js-original-branch" />

<div class="form-actions">
<button name="button" type="button" class="btn btn-success btn-upload-file" id="submit-all"><i aria-hidden="true" data-hidden="true" class="fa fa-spin fa-spinner js-loading-icon hidden"></i>
Replace file
</button><a class="btn btn-cancel" data-dismiss="modal" href="#">Cancel</a>

</div>
</form></div>
</div>
</div>
</div>

</div>
</div>

</div>
</div>
</div>
</div>



</body>
</html>

