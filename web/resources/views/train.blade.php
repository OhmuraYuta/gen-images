<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}" class="dark">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>AI学習室 - Antigravity GenImages</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Inter', 'sans-serif'],
                            display: ['Outfit', 'sans-serif'],
                        },
                    }
                }
            }
        </script>
        <style>
            [x-cloak] { display: none !important; }
            body { background: radial-gradient(circle at top left, #1a1a2e, #16213e, #0f3460); min-height: 100vh; }
            .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
        </style>
        @livewireStyles
    </head>
    <body class="antialiased text-gray-200 py-12 px-4">
        <div class="max-w-4xl mx-auto">
            <!-- Header -->
            <header class="flex justify-between items-center mb-12">
                <div>
                    <h1 class="text-4xl font-bold font-display bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">AI学習室</h1>
                    <p class="text-gray-400 mt-2">先輩だけの専属AIを育てましょう ✨</p>
                </div>
                <a href="/" class="px-6 py-2 glass rounded-full hover:bg-white/10 transition-all font-medium text-sm text-gray-300">
                    ← 生成画面に戻る
                </a>
            </header>

            <!-- Main Content -->
            @livewire('model-trainer')
        </div>

        @livewireScripts
    </body>
</html>
