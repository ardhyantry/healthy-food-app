@extends('layouts.adminSB')
@section('title', 'Quickbite - Edit Menu')

@section('content')
    <div class="container">
        <h2>Edit Menu</h2>

        <form action="{{ route('admin.menus.update', $menu->id) }}" method="POST" enctype="multipart/form-data">
            @csrf
            @method('PUT')

            <!-- Nama -->
            <div class="mb-3">
                <label for="name" class="form-label">Nama Menu</label>
                <input type="text" name="name" class="form-control" value="{{ old('name', $menu->name) }}" required>
            </div>

            <!-- Deskripsi -->
            <div class="mb-3">
                <label for="description" class="form-label">Deskripsi</label>
                <textarea name="description" class="form-control" rows="3" required>{{ old('description', $menu->description) }}</textarea>
            </div>

            <!-- Nutrisi -->
            <div class="mb-3">
                <label for="nutrition_fact" class="form-label">Fakta Nutrisi</label>
                <input type="text" name="nutrition_fact" class="form-control" value="{{ old('nutrition_fact', $menu->nutrition_fact) }}">
            </div>

            <!-- Harga -->
            <div class="mb-3">
                <label for="price" class="form-label">Harga (Rp)</label>
                <input type="number" name="price" class="form-control" value="{{ old('price', $menu->price) }}" required>
            </div>

            <!-- Stok -->
            <div class="mb-3">
                <label for="stock" class="form-label">Stok</label>
                <input type="number" name="stock" class="form-control" value="{{ old('stock', $menu->stock) }}" required>
            </div>

            <!-- Gambar -->
            <div class="mb-3">
                <label for="image_path" class="form-label">Gambar</label>
                <input type="file" name="image_path" class="form-control">
                <br>
                @if($menu->image_path)
                    <img src="{{ asset('storage/' . $menu->image_path) }}" alt="Current Image" width="150">
                @endif
            </div>

            <!-- Kategori -->
            <div class="mb-3">
                <label for="categories_id" class="form-label">Kategori</label>
                <select name="categories_id" class="form-select" required>
                    @foreach($categories as $cat)
                        <option value="{{ $cat->id }}" {{ $cat->id == $menu->categories_id ? 'selected' : '' }}>
                            {{ $cat->name }}
                        </option>
                    @endforeach
                </select>
            </div>

            <!-- Tombol Submit -->
            <button type="submit" class="btn btn-primary">Simpan Perubahan</button>
            <a href="{{ route('admin.menus.index') }}" class="btn btn-secondary">Batal</a>
        </form>
    </div>
@endsection