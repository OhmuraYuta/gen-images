<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI Image Generator</title>
        <link rel="preconnect" href="https://fonts.bunny.net">
        <link href="https://fonts.bunny.net/css?family=outfit:400,500,600,700" rel="stylesheet" />
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Outfit', sans-serif; }
            .glass {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .dark .glass {
                background: rgba(26, 26, 21, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
        </style>
        @livewireStyles
    </head>
    <body class="bg-[#FDFDFC] dark:bg-[#0a0a0a] text-[#1b1b18] dark:text-[#EDEDEC] min-h-screen">
        <div class="relative overflow-hidden min-h-screen">
            <!-- Background Decorations -->
            <div class="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
                <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#f53003] opacity-10 filter blur-[120px] rounded-full animate-pulse"></div>
                <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500 opacity-10 filter blur-[120px] rounded-full animate-pulse" style="animation-delay: 1s;"></div>
            </div>

            <nav class="flex items-center justify-between px-8 py-6 w-full lg:max-w-7xl mx-auto">
                <div class="flex items-center gap-2">
                    <div class="p-2 bg-[#1b1b18] dark:bg-[#eeeeec] rounded-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6 text-white dark:text-[#1c1c1a]">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                        </svg>
                    </div>
                    <span class="text-xl font-bold tracking-tight">AI Studio</span>
                </div>
                <div class="flex items-center gap-4">
                    <a href="/train" class="px-4 py-2 glass rounded-xl hover:bg-white/10 transition-all font-medium text-sm flex items-center group">
                        <span class="mr-2 group-hover:rotate-12 transition-transform">🎓</span> AI学習室
                    </a>
                </div>
            </nav>

            <main class="lg:max-w-7xl mx-auto px-8 py-12">
                <livewire:image-generator />
            </main>
        </div>
        @livewireScripts
    </body>
</html>